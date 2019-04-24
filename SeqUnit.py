#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:37
# @Author  : Tianyu Liu

import os
import tensorflow as tf

from AttentionUnit import AttentionWrapper
from dualAttentionUnit import dualAttentionWrapper
from LstmUnit import LstmUnit
from fgateLstmUnit import fgateLstmUnit
from mlpUnit import mlpUnit
from model import *


class SeqUnit(object):
    def __init__(self, batch_size, hidden_size, emb_size, field_size, pos_size, source_vocab, field_vocab,
                 position_vocab, target_vocab, field_concat, position_concat, fgate_enc, dual_att,
                 encoder_add_pos, decoder_add_pos, learning_rate, scope_name, name, use_coverage, coverage_penalty, 
                 fieldid2word, copy_gate_penalty, use_copy_gate, gpt_hparams, gpt_out_mask, vocab_ind,
                 empty_token=28920, stop_token=50256, max_length=85, encoder_type='mlp'):
        '''
        batch_size, hidden_size, emb_size, field_size, pos_size: size of batch; hidden layer; word/field/position embedding
        source_vocab, target_vocab, field_vocab, position_vocab: vocabulary size of encoder words; decoder words; field types; position
        field_concat, position_concat: bool values, whether concat field/position embedding to word embedding for encoder inputs or not
        fgate_enc, dual_att: bool values, whether use field-gating / dual attention or not
        encoder_add_pos, decoder_add_pos: bool values, whether add position embedding to field-gating encoder / decoder with dual attention or not

        ###
        original full vocab ind
        empty_token=28920, stop_token=50256
        '''

        # data options
        self.empty_token = empty_token
        self.stop_token = stop_token
        self.max_length = max_length
        self.start_token = empty_token
        self.select_ind = vocab_ind
        self.gpt_out_mask = gpt_out_mask
        self.fieldid2word = fieldid2word

        # model hyperparams
        self.gpt_hparams = gpt_hparams
        self.hidden_size = self.gpt_hparams.n_embd

        # model architecture options
        self.use_coverage = use_coverage
        self.coverage_penalty = coverage_penalty
        self.use_copy_gate = use_copy_gate
        self.copy_gate_penalty = copy_gate_penalty
        self.fgate_enc = fgate_enc
        self.dual_att = dual_att
        self.scope_name = scope_name
        self.name = name
        self.encoder_type = encoder_type

        # embedding sizes
        self.emb_size = self.gpt_hparams.n_embd # word embedding size
        self.field_size = field_size # field embedding size
        self.pos_size = pos_size # position embedding size
        self.field_concat = field_concat
        self.position_concat = position_concat
        self.encoder_add_pos = encoder_add_pos
        self.decoder_add_pos = decoder_add_pos
        self.uni_size = self.emb_size if not field_concat else self.emb_size+field_size
        self.uni_size = self.uni_size if not position_concat else self.uni_size+2*pos_size
        self.field_encoder_size = field_size if not encoder_add_pos else field_size+2*pos_size
        self.field_attention_size = field_size if not decoder_add_pos else field_size+2*pos_size
        self.dec_input_size = self.emb_size + field_size + 2 * pos_size # FIXME not conditioned?

        # source and target vocabulary sizes, field and position vocabulary sizes
        self.source_vocab = self.gpt_hparams.n_vocab
        self.target_vocab = self.gpt_hparams.n_vocab
        self.field_vocab = field_vocab
        self.position_vocab = position_vocab

        # training options
        self.grad_clip = 5.0

        self.units = {}
        self.params = {}

        self.define_input_placeholders()

        self.define_encoder_unit()

        context_outputs = self.define_decoder_arch()

        # get GPT embeddings
        self.lookup_all_embeddings()

        self.define_encoder_arch()

        # attention and copy layers
        if self.dual_att:
            print('dual attention mechanism used')
            with tf.variable_scope(scope_name):
                self.att_layer = dualAttentionWrapper(self.dec_input_size, self.hidden_size, self.hidden_size, self.field_attention_size, "attention")
                self.units.update({'attention': self.att_layer})
        else:
            print("normal attention used")
            with tf.variable_scope(scope_name):
                self.att_layer = AttentionWrapper(self.hidden_size, self.hidden_size, self.en_outputs, "attention")
                self.units.update({'attention': self.att_layer})

        # loss functions
        # calculate those locations where field values are present
        self.copy_gate_mask = tf.cast(
                        tf.greater(self.decoder_pos_input, tf.zeros_like(self.decoder_pos_input)), tf.float32)
        self.copy_gate_mask = tf.concat([self.copy_gate_mask, tf.zeros([tf.shape(self.encoder_input)[0], 1], tf.float32)], 1)

        # decoder for training
        # get start values to start gpt generation
        logits0 = context_outputs['logits'][:, -1, :]
        dist0 = tf.nn.softmax(logits0) # start token
        x0 = tf.cast(tf.argmax(dist0, 1), tf.int32)
        past0 = context_outputs['presents']
        hidden0 = context_outputs['hidden'][:, -1, :]

        de_outputs, _, self.de_conv_loss, self.copy_gate_loss = self.decoder_t(self.decoder_input, self.decoder_len, x0, past0, hidden0)

        # decoder for testing
        self.g_tokens, self.atts = self.decoder_g(x0, past0, hidden0)

        ### enc-dec loss
        self.decoder_output_one_hot = tf.one_hot(indices=self.decoder_output, 
                                                depth=self.target_vocab,
                                                axis=-1)

        # mask for dec. plus eos
        dec_shape_len = tf.shape(self.decoder_output)[1]
        batch_nums = tf.range(0, dec_shape_len)
        batch_nums = tf.expand_dims(batch_nums, 0)
        batch_nums = tf.tile(batch_nums, [self.batch_size, 1])
        decoder_len_com = tf.expand_dims(self.decoder_len, 1)
        decoder_len_com = tf.tile(decoder_len_com, [1, dec_shape_len])
        mask = tf.cast(
                tf.less_equal(batch_nums, decoder_len_com), tf.float32)

        # total loss
        losses = -tf.reduce_sum(self.decoder_output_one_hot * tf.log(de_outputs + 1e-6), 2)
        losses = mask * losses

        # faster. original reduce mean
        self.mean_loss = tf.reduce_sum(losses)

        self.de_conv_loss *= self.coverage_penalty

        self.copy_gate_loss = self.copy_gate_penalty * tf.reduce_sum(self.copy_gate_loss)

        if self.use_copy_gate:
            self.mean_loss += self.copy_gate_loss

        if self.use_coverage:
            self.mean_loss += self.de_conv_loss

        train_params = tf.trainable_variables()

        # train enc-dec
        with tf.variable_scope(scope_name):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, train_params, colocate_gradients_with_ops=True), self.grad_clip)

        # accumulate gradient
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.acc_gradients = list(map(lambda param: tf.get_variable(param.name.split(":")[0],
                                                                    param.get_shape(), param.dtype,
                                                                    tf.constant_initializer(0.0), trainable=False),
                                                                    train_params))

            # initialize losses?
            self._loss = tf.get_variable("acc_loss", (), tf.float32, tf.constant_initializer(0.0), trainable=False)
            self._cov_loss = tf.get_variable("acc_cov_loss", (), tf.float32, tf.constant_initializer(0.0), trainable=False)
            self._gate_loss = tf.get_variable("acc_gate_loss", (), tf.float32, tf.constant_initializer(0.0), trainable=False)

            self.accumulate_gradients()

            # train update
            self.update = self.opt.apply_gradients(
                zip(list(map(lambda v: v.value(), self.acc_gradients)), train_params), global_step=self.global_step)

            # collect all values to reset after updating with accumulated gradient
            self.reset = list(map(lambda param: param.initializer, self.acc_gradients))
            self.reset.append(self._loss.initializer)
            self.reset.append(self._cov_loss.initializer)
            self.reset.append(self._gate_loss.initializer)

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def accumulate_gradients(self):
        # We abuse the gradient descent optimizer for accumulating gradients and loss (summing)
        acc_opt = tf.train.GradientDescentOptimizer(-1.0)
        self.accumulate_gradients = acc_opt.apply_gradients(zip(self.grads, self.acc_gradients))
        self.acc_loss = acc_opt.apply_gradients([(self.mean_loss, self._loss)])
        self.acc_cov_loss = acc_opt.apply_gradients([(self.de_conv_loss, self._cov_loss)])
        self.acc_gate_loss = acc_opt.apply_gradients([(self.copy_gate_loss, self._gate_loss)])

    def define_input_placeholders(self):
        """
        define all placeholders
        Returns:
            None
        """
        self.gpt_context = tf.placeholder(tf.int32, [None, None])
        self.encoder_input = tf.placeholder(tf.int32, [None, None])
        self.encoder_field = tf.placeholder(tf.int32, [None, None])
        self.encoder_pos = tf.placeholder(tf.int32, [None, None])
        self.encoder_rpos = tf.placeholder(tf.int32, [None, None])
        self.decoder_input = tf.placeholder(tf.int32, [None, None])
        self.encoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_output = tf.placeholder(tf.int32, [None, None])
        self.enc_mask = tf.sign(tf.to_float(self.encoder_pos))
        self.decoder_field_input = tf.placeholder(tf.int32, [None, None])
        self.decoder_pos_input = tf.placeholder(tf.int32, [None, None])
        self.decoder_rpos_input = tf.placeholder(tf.int32, [None, None])
        self.context = tf.placeholder(tf.int32, [None, None])
        self.encoder_input_real = tf.placeholder(tf.int32, [None, None])
        self.decoder_input_real = tf.placeholder(tf.int32, [None, None])

        return

    def define_encoder_unit(self):
        """
        define LSTM encoder unit
        Returns:
            None
        """
        with tf.variable_scope(self.scope_name):
            if self.encoder_type == 'mlp':
                self.enc_lstm = mlpUnit(self.hidden_size, self.uni_size,
                                        'encoder_select')
            else:
                if self.fgate_enc:
                    print('field-gated encoder LSTM')
                    self.enc_lstm = fgateLstmUnit(self.hidden_size, self.uni_size,
                                                  self.field_encoder_size, 'encoder_select')
                else:
                    print('normal encoder LSTM')
                    self.enc_lstm = LstmUnit(self.hidden_size, self.uni_size, 'encoder_lstm')

        self.units.update({'encoder_lstm': self.enc_lstm})
        return

    def define_encoder_arch(self):
        if self.encoder_type == "mlp":
            self.en_outputs, en_state = self.fgate_encoder(self.encoder_embed, self.field_pos_embed,
                                                           self.encoder_len)  # plus domain embedding
        else:
            if self.fgate_enc:
                print('field gated encoder used')
                self.en_outputs, en_state = self.fgate_encoder(self.encoder_embed, self.field_pos_embed, self.encoder_len) # plus domain embedding
            else:
                print('normal encoder used')
                self.en_outputs, en_state = self.encoder(self.encoder_embed, self.encoder_len) #FIXME where is this

    def define_decoder_arch(self):
        # define GPT decoder
        self.batch_size = tf.shape(self.decoder_input)[0]
        # initialize embedding
        gpt_emb_init_tune('model', self.gpt_hparams)
        # combine decoder contexts
        self.gpt_context_in = tf.concat([self.context, self.gpt_context], 1)
        context_outputs = self.step_gpt(self.gpt_hparams, self.gpt_context_in, self.batch_size)
        return context_outputs

    def lookup_all_embeddings(self):
        with tf.variable_scope('model', reuse=True):
            ### use the one in gpt2
            self.embedding = tf.get_variable('wte_tune', [self.gpt_hparams.n_vocab, self.gpt_hparams.n_embd], trainable=False)

        # look up and combine embeddings
        with tf.device("/gpu:1"):
            with tf.variable_scope(self.scope_name):
                self.field_id2word = tf.constant(self.fieldid2word)
                self.gpt_out_mask = tf.constant(self.gpt_out_mask)

                self.encoder_embed = tf.nn.embedding_lookup(self.embedding, self.encoder_input)
                self.decoder_embed = tf.nn.embedding_lookup(self.embedding, self.decoder_input)

                if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos: # True
                    self.field_word = tf.nn.embedding_lookup(self.field_id2word, self.encoder_field) # batch * enc_len * 3
                    self.field_embed = tf.reduce_mean(
                                        tf.nn.embedding_lookup(self.embedding, self.field_word), 2)
                    self.field_pos_embed = self.field_embed

                if self.position_concat or self.encoder_add_pos or self.decoder_add_pos: # True
                    self.pembedding = tf.get_variable('pembedding', [self.position_vocab, self.pos_size])
                    self.rembedding = tf.get_variable('rembedding', [self.position_vocab, self.pos_size])
                    self.pos_embed = tf.nn.embedding_lookup(self.pembedding, self.encoder_pos)
                    self.rpos_embed = tf.nn.embedding_lookup(self.rembedding, self.encoder_rpos)
                    if self.encoder_add_pos or self.decoder_add_pos: # True
                        self.field_pos_embed = tf.concat([self.field_embed, self.pos_embed, self.rpos_embed], 2)

                self.field_word_dec = tf.nn.embedding_lookup(self.field_id2word, self.decoder_field_input) # batch * dec_len * 3
                self.field_embed_dec = tf.reduce_mean(
                                        tf.nn.embedding_lookup(self.embedding, self.field_word_dec), 2)

                self.pos_embed_dec = tf.nn.embedding_lookup(self.pembedding, self.decoder_pos_input)
                self.rpos_embed_dec = tf.nn.embedding_lookup(self.rembedding, self.decoder_rpos_input)

                ### decoder plus start token
                self.decoder_field_pos_emb = tf.concat([self.field_embed_dec, self.pos_embed_dec, self.rpos_embed_dec], 2)
                field_pos_embed_size = tf.shape(self.decoder_field_pos_emb)[2]
                field_pos_embed_zeros = tf.zeros([self.batch_size, 1, field_pos_embed_size])
                self.decoder_field_pos_emb = tf.concat([field_pos_embed_zeros, self.decoder_field_pos_emb], 1) # dec_len + 1

        # FIXME why are we tracking this
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            self.params.update({'pembedding': self.pembedding})
            self.params.update({'rembedding': self.rembedding})

    def fgate_encoder(self, inputs, fields, inputs_len):
        batch_size = tf.shape(self.encoder_input)[0]
        max_time = tf.shape(self.encoder_embed)[1]
        hidden_size = self.hidden_size

        time = tf.constant(0, dtype=tf.int32)
        h0 = (tf.zeros([batch_size, hidden_size], dtype=tf.float32),
              tf.zeros([batch_size, hidden_size], dtype=tf.float32))
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0,2]))
        fields_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        fields_ta = fields_ta.unstack(tf.transpose(fields, [1,0,2]))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, d_t, s_t, emit_ta, finished):
            o_t, s_nt = self.enc_lstm(x_t, d_t, s_t, finished)
            emit_ta = emit_ta.write(t, o_t)
            finished = tf.greater_equal(t+1, inputs_len)
            x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.uni_size], dtype=tf.float32),
                                     lambda: inputs_ta.read(t+1))
            d_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.field_attention_size], dtype=tf.float32),
                                     lambda: fields_ta.read(t+1))
            return t+1, x_nt, d_nt, s_nt, emit_ta, finished

        _, _, _, state, emit_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, _5, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, inputs_ta.read(0), fields_ta.read(0), h0, emit_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        return outputs, state

    def step_gpt(self, hparams, tokens, batch_size, past=None):
        """
        GPT2 model is imported here, as defined in model.py
        Args:
            hparams: Input parameters of the GPT architecture
            tokens: input tokens
            batch_size: batch size
            past: #TODO

        Returns: Output of transformer - logits in output sequence

        """

        lm_output = model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        hidden = lm_output['hidden']
        presents.set_shape(past_shape(hparams=hparams, batch_size=batch_size))
        return {'logits': logits, 'presents': presents, 'hidden': hidden}

    def decoder_t(self, inputs, inputs_len, x0, past0, hidden0):
        """
        Decoder for training
        Args:
            inputs: ground truth inputs
            inputs_len: length of ground truth input
            x0: #TODO
            past0: #TODO
            hidden0: #TODO

        Returns:

        """
        # gather p_gen and att_weights
        batch_size = tf.shape(self.decoder_input)[0]
        max_time = tf.shape(self.decoder_input)[1]
        encoder_len = tf.shape(self.encoder_embed)[1]

        time = tf.constant(0, dtype=tf.int32)
        f0 = tf.zeros([batch_size], dtype=tf.bool)

        inputs_ta = tf.TensorArray(dtype=tf.int32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0]))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
        emit_gate = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        # coverage mechanism
        coverage_att_sum = tf.zeros([batch_size, encoder_len], dtype=tf.float32)
        covloss0 = 0.0

        def loop_fn(t, x_t, past, hidden, emit_ta, emit_gate, coverage_att_sum, covloss, finished):
            """
            Decoding loop
            Args:
                t: sequence index
                x_t: input at location t
                past: decoded string so far
                hidden: #TODO
                emit_ta: TODO
                emit_gate:  TODO
                coverage_att_sum: TODO
                covloss: TODO
                finished: TODO

            Returns:

            """
            # gpt generate
            temperature = 1.0  # hard coded temperature or noise in GPT logit output
            next_outputs = self.step_gpt(self.gpt_hparams, x_t[:, tf.newaxis], self.batch_size, past=past)
            logits = next_outputs['logits'][:, -1, :] / tf.to_float(temperature)
            o_dist = tf.nn.softmax(logits)
            past_nt = tf.concat([past, next_outputs['presents']], axis=-2)
            hidden_nt = next_outputs['hidden'][:, -1, :]

            # concat field pos embedding
            batch_nums_time = tf.range(0, limit=batch_size)
            time_batch = tf.fill([batch_size], t)
            collect_ind = tf.stack([batch_nums_time, time_batch], axis=1)
            this_field_pos_emb = tf.gather_nd(self.decoder_field_pos_emb, collect_ind) # [batch_size, field + pos]
            att_x_in = tf.nn.embedding_lookup(self.embedding, x_t)
            att_x_in = tf.concat([att_x_in, this_field_pos_emb], axis=1)

            # pass the hidden weights into the attention layer to get
            # gen gate probability
            o_weight, p_gen = self.att_layer(hidden_nt, att_x_in, hidden, coverage_att_sum,
                                             self.en_outputs, self.field_pos_embed, finished=finished)

            # generative probabilty is weighted product of gen gate probability and gpt softmax
            out_dist = p_gen * o_dist # batch * self.target_vocab

            # project pointer output logits into target vocabulary
            att_dist = o_weight
            batch_nums = tf.range(0, limit=batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)
            batch_nums = tf.tile(batch_nums, [1, encoder_len])
            indices = tf.stack((batch_nums, self.encoder_input), axis=2) # batch_size * enc_len * 2
            shape = [batch_size, self.target_vocab]
            attn_dists_projected = tf.scatter_nd(indices, att_dist, shape)

            # combine both weighted probabilities
            final_dists = out_dist + (1 - p_gen) * attn_dists_projected

            # consider only those locations with field values in the output
            copy_mask = tf.gather_nd(self.copy_gate_mask, collect_ind)

            # write to tensor array
            emit_gate = emit_gate.write(t, tf.multiply(p_gen, copy_mask))
            emit_ta = emit_ta.write(t, final_dists)

            this_covloss = tf.reduce_sum(tf.minimum(coverage_att_sum, o_weight))
            covloss += this_covloss
            coverage_att_sum += o_weight

            # stop condition
            finished = tf.greater_equal(t, inputs_len)
            x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.fill([batch_size], self.stop_token),
                                     lambda: inputs_ta.read(t))

            return t+1, x_nt, past_nt, hidden_nt, emit_ta, emit_gate, coverage_att_sum, covloss, finished

        _, _, past_final, hidden_final, emit_ta, emit_gate, coverage_att_sum, emit_covloss, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, _5, _6, _7, _8, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, x0, past0, hidden0, emit_ta, emit_gate, coverage_att_sum, covloss0, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        outputs_gate = tf.squeeze(tf.transpose(emit_gate.stack(), [1,0,2]))
        return outputs, past_final, emit_covloss, outputs_gate

    def decoder_g(self, x0, past0, hidden0):
        """
        Decoder for generation
        Args:
            x0: data
            past0: ?
            hidden0: ?

        Returns:

        """
        batch_size = tf.shape(self.encoder_input)[0]
        encoder_len = tf.shape(self.encoder_embed)[1]

        time = tf.constant(0, dtype=tf.int32)
        f0 = tf.zeros([batch_size], dtype=tf.bool)

        # concat with field + pos + rpos to input
        x0_field = tf.nn.embedding_lookup(self.embedding, tf.fill([batch_size], self.empty_token))
        x0_pos = tf.nn.embedding_lookup(self.pembedding, tf.zeros([batch_size], dtype=tf.int32))
        x0_rpos = tf.nn.embedding_lookup(self.rembedding, tf.zeros([batch_size], dtype=tf.int32))
        field_pos0 = tf.concat([x0_field, x0_pos, x0_rpos], 1)

        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
        att_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        # coverage mechanisim
        coverage_att_sum = tf.zeros([batch_size, encoder_len], dtype=tf.float32)

        att_mask = tf.ones([batch_size, encoder_len], dtype=tf.float32)

        def loop_fn(t, x_t, past, hidden, field_pos_emb, emit_ta, att_ta, coverage_att_sum, att_mask, finished):
            """

            Args:
                t:
                x_t:
                past:
                hidden:
                field_pos_emb:
                emit_ta:
                att_ta:
                coverage_att_sum:
                att_mask:
                finished:

            Returns:

            """
            # gpt generate
            temperature = 1.0  # hard coded temperature or noise in GPT logit output
            next_outputs = self.step_gpt(self.gpt_hparams, x_t[:, tf.newaxis], self.batch_size, past=past)
            logits = next_outputs['logits'][:, -1, :] / tf.to_float(temperature)
            o_dist = tf.nn.softmax(logits)
            past_nt = tf.concat([past, next_outputs['presents']], axis=-2)
            hidden_nt = next_outputs['hidden'][:, -1, :]

            # concat field pos embedding
            att_x_in = tf.nn.embedding_lookup(self.embedding, x_t)
            att_x_in = tf.concat([att_x_in, field_pos_emb], axis=1)

            # pass the hidden weights into the attention layer to get
            # gen gate probability
            o_weight, p_gen = self.att_layer(hidden_nt, att_x_in, hidden, coverage_att_sum,
                                             self.en_outputs, self.field_pos_embed, finished=finished)

            # generative probabilty is weighted product of gen gate probability and gpt softmax
            out_dist = p_gen * o_dist # batch * self.target_vocab


            ### mask previous
            att_dist = (1 - p_gen) * o_weight  # batch * len
            att_dist *= att_mask
            batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
            batch_nums = tf.tile(batch_nums, [1, encoder_len]) # shape (batch_size, enc_len)
            indices = tf.stack((batch_nums, self.encoder_input), axis=2) # shape (batch_size, enc_len, 2)
            shape = [batch_size, self.target_vocab]
            attn_dists_projected = tf.scatter_nd(indices, att_dist, shape) # batch * target_vocab

            # combine both weighted probabilities
            final_dists = out_dist + attn_dists_projected

            # coverage
            coverage_att_sum += o_weight

            # write to tensor array
            emit_ta = emit_ta.write(t, final_dists)
            att_ta = att_ta.write(t, tf.transpose(o_weight, [1,0]))

            x_nt = tf.cast(tf.argmax(final_dists, 1), tf.int32)

            # field pos emb next round
            next_token_att = tf.cast(tf.argmax(attn_dists_projected, 1), tf.int32)
            mask = tf.cast(tf.equal(x_nt, next_token_att), tf.int32)

            att_pos = tf.cast(tf.argmax(att_dist, 1), tf.int32)
            batch_num = tf.range(0, limit=batch_size)
            this_dec_indices = tf.stack([batch_num, att_pos], axis=1) # batch_size * 2
            this_dec_field_id = tf.gather_nd(self.encoder_field, this_dec_indices)
            this_dec_pos_id = tf.gather_nd(self.encoder_pos, this_dec_indices)
            this_dec_rpos_id = tf.gather_nd(self.encoder_rpos, this_dec_indices)

            this_dec_field_id = this_dec_field_id * mask
            this_dec_pos_id = this_dec_pos_id * mask
            this_dec_rpos_id = this_dec_rpos_id * mask

            this_dec_field_word = tf.nn.embedding_lookup(self.field_id2word, this_dec_field_id) # batch * 3
            this_dec_field_emb = tf.reduce_mean(
                                tf.nn.embedding_lookup(self.embedding, this_dec_field_word), 1) # batch_size * field_emb_size

            this_dec_pos_emb = tf.nn.embedding_lookup(self.pembedding, this_dec_pos_id)
            this_dec_rpos_emb = tf.nn.embedding_lookup(self.rembedding, this_dec_rpos_id)

            field_pos_nt = tf.concat([this_dec_field_emb, this_dec_pos_emb, this_dec_rpos_emb], 1)

            # mask atten pos of previous
            att_pos *= mask
            att_pos_tile = tf.expand_dims(att_pos, 1)
            att_pos_tile = tf.tile(att_pos_tile, [1, encoder_len])
            att_mask_enc = tf.range(0, encoder_len)
            att_mask_enc = tf.expand_dims(att_mask_enc, 0)
            att_mask_enc = tf.tile(att_mask_enc, [batch_size, 1])
            mask_enc = tf.cast(tf.not_equal(att_pos_tile, att_mask_enc), tf.float32)
            att_mask *= mask_enc

            finished = tf.logical_or(finished, tf.equal(x_nt, self.stop_token))
            finished = tf.logical_or(finished, tf.greater_equal(t, self.max_length))
            return t+1, x_nt, past_nt, hidden_nt, field_pos_nt, emit_ta, att_ta, coverage_att_sum, att_mask, finished

        _, _, past_final, hidden_final, field_pos_nt, emit_ta, att_ta, _, _, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, _5, _6, _7, _8, _9, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, x0, past0, hidden0, field_pos0, emit_ta, att_ta, coverage_att_sum, att_mask, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        pred_tokens = tf.argmax(outputs, 2)
        atts = att_ta.stack()
        return pred_tokens, atts

    def create_feed_dict(self, x, training=True):
        """
        Create feed dict with placeholder keys for feeding x input to model
        Args:
            x: dict, input
            training: bool, for training or inference

        Returns:
            feed_dict
        """
        feed_dict = {self.encoder_input: x['enc_in'], self.encoder_field: x['enc_fd'],
                     self.encoder_len: x['enc_len'], self.encoder_pos: x['enc_pos'],
                     self.encoder_rpos: x['enc_rpos'], self.decoder_input: x['dec_in'],

                     self.gpt_context: x['gpt_context'], self.context: x['context'],
                     self.encoder_input_real: x['enc_in_real'],
                     self.decoder_input_real: x['dec_in_real']}
        if training:
            feed_dict.update({self.decoder_len: x['dec_len'], self.decoder_output: x['dec_out'],
                         self.decoder_field_input: x['dec_field'],
                         self.decoder_pos_input: x['dec_pos'],
                         self.decoder_rpos_input: x['dec_rpos']})
        else:
            pass
        return feed_dict

    def __call__(self, x, sess, mode):
        """
        Calling this instance either accumulates gradients or runs optimizer update
        Args:
            x: data
            sess: TF Session
            mode: 0/1 accumulate gradient/run opt update
        Returns:
            total loss, copy gate loss, ?, ?
        """
        if mode == 0:
            feed_dict = self.create_feed_dict(x, training=True)
            loss, copy_gate_loss, de_conv_loss, _, _, _, _ = sess.run([self.mean_loss,
                                                                       self.copy_gate_loss,
                                                                       self.de_conv_loss,
                                                                       self.accumulate_gradients,
                                                                       self.acc_loss,
                                                                       self.acc_cov_loss,
                                                                       self.acc_gate_loss],
                                                                       feed_dict=feed_dict)
            return loss, copy_gate_loss, de_conv_loss, 0

        if mode == 1:
            acc_loss, acc_cov_loss, acc_gate_loss = sess.run([self._loss, self._cov_loss, self._gate_loss])
            sess.run(self.update)
            sess.run(self.reset)

            return acc_loss, acc_gate_loss, acc_cov_loss

    def generate(self, x, sess):
        """
        Generate predictions given input
        Args:
            x: input data
            sess: TF Session

        Returns:
            predictions and ? #TODO
        """
        feed_dict = self.create_feed_dict(x, training=False)
        predictions, atts = sess.run([self.g_tokens, self.atts], feed_dict=feed_dict)
        return predictions, atts

    def save(self, path, sess):
        """
        Save model to file
        Args:
            path: path to save file
            sess: TF Session

        Returns:
            None
        """
        checkpoint_path = os.path.join(path, "wiki2bio_model.ckpt")
        self.saver.save(sess, checkpoint_path, global_step=self.global_step.eval())
        print("Model saved on global step %d." % (self.global_step.eval()))
        return

    def load(self, path, sess):
        """
        Load saved model from checkpoint
        Args:
            path: checkpoint path
            sess: TF session

        Returns:
            None
        """
        ckpt = tf.train.get_checkpoint_state(path)
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)
        return


















