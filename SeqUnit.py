#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:37
# @Author  : Tianyu Liu

import sys
import os
import tensorflow as tf
import pickle
from AttentionUnit import AttentionWrapper
from dualAttentionUnit import dualAttentionWrapper
from LstmUnit import LstmUnit
from fgateLstmUnit import fgateLstmUnit
from OutputUnit import OutputUnit
from OutputUnit_gpt import OutputUnit_gpt

from model import *


class SeqUnit(object):
    def __init__(self, batch_size, hidden_size, emb_size, field_size, pos_size, source_vocab, field_vocab,
                 position_vocab, target_vocab, field_concat, position_concat, fgate_enc, dual_att,
                 encoder_add_pos, decoder_add_pos, learning_rate, scope_name, name, use_coverage, coverage_penalty, 
                 fieldid2word, copy_gate_penalty, use_copy_gate, gpt_hparams, gpt_out_mask, vocab_ind, empty_token=5713, stop_token=6975, max_length=85):


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

        ### caution dynamic batch size
        # self.batch_size = batch_size
        self.gpt_hparams = gpt_hparams
        self.hidden_size = self.gpt_hparams.n_embd

        # self.emb_size = emb_size
        self.emb_size = self.gpt_hparams.n_embd

        self.field_size = field_size
        self.pos_size = pos_size
        self.uni_size = self.emb_size if not field_concat else self.emb_size+field_size
        self.uni_size = self.uni_size if not position_concat else self.uni_size+2*pos_size
        self.field_encoder_size = field_size if not encoder_add_pos else field_size+2*pos_size
        self.field_attention_size = field_size if not decoder_add_pos else field_size+2*pos_size
        self.source_vocab = self.gpt_hparams.n_vocab
        self.target_vocab = self.gpt_hparams.n_vocab
        self.field_vocab = field_vocab
        self.position_vocab = position_vocab
        self.grad_clip = 5.0
        self.empty_token = empty_token
        self.stop_token = stop_token
        self.max_length = max_length
        self.scope_name = scope_name
        self.name = name
        self.field_concat = field_concat
        self.position_concat = position_concat
        self.fgate_enc = fgate_enc
        self.dual_att = dual_att
        self.encoder_add_pos = encoder_add_pos
        self.decoder_add_pos = decoder_add_pos

        self.use_coverage = use_coverage
        self.coverage_penalty = coverage_penalty

        self.use_copy_gate = use_copy_gate
        self.copy_gate_penalty = copy_gate_penalty

        self.dec_input_size = self.emb_size + field_size + 2 * pos_size

        self.gpt_out_mask = gpt_out_mask
        self.start_token = empty_token
        self.select_ind = vocab_ind

        self.units = {}
        self.params = {}


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

        ### add field pos rpos to decoder
        self.decoder_field_input = tf.placeholder(tf.int32, [None, None])
        self.decoder_pos_input = tf.placeholder(tf.int32, [None, None])
        self.decoder_rpos_input = tf.placeholder(tf.int32, [None, None])

        self.context = tf.placeholder(tf.int32, [None, None])


        with tf.variable_scope(scope_name):
            if self.fgate_enc:
                print ('field-gated encoder LSTM')
                self.enc_lstm = fgateLstmUnit(self.hidden_size, self.uni_size, self.field_encoder_size, 'encoder_select')
            else:
                print ('normal encoder LSTM')
                self.enc_lstm = LstmUnit(self.hidden_size, self.uni_size, 'encoder_lstm')
            # self.dec_lstm = LstmUnit(self.hidden_size, self.emb_size, 'decoder_lstm')
            # self.dec_lstm = LstmUnit(self.hidden_size, self.dec_input_size, 'decoder_lstm')
            self.dec_out = OutputUnit(self.hidden_size, self.target_vocab, 'decoder_output')
            self.dec_out_lstm = OutputUnit(self.hidden_size, self.target_vocab, 'decoder_output_lstm')

            # self.hidden_w = tf.get_variable('hidden_w', [2*self.hidden_size, self.hidden_size])
            # self.hidden_b = tf.get_variable('hidden_b', [self.hidden_size])


            # self.gpt_out_mask = tf.sigmoid(
            #                     tf.get_variable('gpt_out_mask', shape=[self.target_vocab], initializer=tf.zeros_initializer, trainable=True))


        self.units.update({'encoder_lstm': self.enc_lstm})

        self.batch_size = tf.shape(self.decoder_input)[0]



        gpt_emb_init_tune('model', self.gpt_hparams, tf.constant(self.select_ind))

        # with tf.device("/gpu:1"):
        ### start with context tokens of domain
        ### start reuse of gpt
        # context_outputs = self.step_gpt(self.gpt_hparams, self.gpt_context, self.batch_size)
        self.gpt_context_in = tf.concat([self.encoder_input, self.gpt_context], 1)
        context_outputs = self.step_gpt(self.gpt_hparams, self.gpt_context_in, self.batch_size)
        # context_outputs = self.step_gpt(self.gpt_hparams, self.encoder_input, self.batch_size)
        logits0 = context_outputs['logits'][:, -1, :]
        dist0 = tf.nn.softmax(logits0) # start token
        x0 = tf.cast(tf.argmax(dist0, 1), tf.int32)
        past0 = context_outputs['presents']
        hidden0 = context_outputs['hidden'][:, -1, :]

        # gpt_emb_init('model', self.gpt_hparams)

        # ======================================== embeddings ======================================== #

        with tf.variable_scope('model', reuse=True):
            ### use the one in gpt2
            self.embedding = tf.get_variable('wte_tune', [self.gpt_hparams.n_vocab, self.gpt_hparams.n_embd], trainable=False)


            # self.en_outputs = context_outputs['hidden']


        # with tf.variable_scope(scope_name):
        #     proj_init = tf.transpose(self.embedding, [1,0])
        #     self.dec_out = OutputUnit_gpt(self.hidden_size, self.target_vocab, proj_init, 'decoder_output')





    # with tf.device("/gpu:1"):
        with tf.variable_scope(scope_name):

            self.field_id2word = tf.constant(fieldid2word)
            self.gpt_out_mask = tf.constant(gpt_out_mask)

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


            ### remove encoder
            # self.en_outputs = tf.concat([self.encoder_embed, self.field_embed, self.pos_embed, self.rpos_embed], 2)
            # self.en_outputs = tf.layers.dense(self.en_outputs, self.hidden_size)


        # if self.field_concat or self.fgate_enc:
        #     self.params.update({'fembedding': self.fembedding})
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            self.params.update({'pembedding': self.pembedding})
            self.params.update({'rembedding': self.rembedding})

        # ======================================== encoder ======================================== #
        if self.fgate_enc:
            print ('field gated encoder used')
            self.en_outputs, en_state = self.fgate_encoder(self.encoder_embed, self.field_pos_embed, self.encoder_len) # plus domain embedding
        else:
            print ('normal encoder used')
            en_outputs, en_state = self.encoder(self.encoder_embed, self.encoder_len)

        # ### try with gpt out as en out
        # self.en_outputs = tf.concat([self.en_outputs, self.field_pos_embed], 2)


        # ### remove encoder
        # self.en_outputs = tf.concat([self.encoder_embed, self.field_embed, self.pos_embed, self.rpos_embed], 2)
        # self.en_outputs = tf.layers.dense(self.en_outputs, self.hidden_size)



        # ======================================== decoder ======================================== #

        if self.dual_att:
            print ('dual attention mechanism used')
            with tf.variable_scope(scope_name):
                # self.att_layer = dualAttentionWrapper(self.emb_size, self.hidden_size, self.hidden_size, self.field_attention_size,
                #                                         en_outputs, self.field_pos_embed, "attention")
                self.att_layer = dualAttentionWrapper(self.dec_input_size, self.hidden_size, self.hidden_size, self.field_attention_size, "attention")
                self.units.update({'attention': self.att_layer})
        else:
            print ("normal attention used")
            with tf.variable_scope(scope_name):
                self.att_layer = AttentionWrapper(self.hidden_size, self.hidden_size, en_outputs, "attention")
                self.units.update({'attention': self.att_layer})
                # self.att_layer_rnet = AttentionWrapper(self.hidden_size, self.hidden_size, en_outputs_f, "attention_rnet")
                # self.units.update({'attention_rnet': self.att_layer_rnet})



        self.copy_gate_mask = tf.cast(
                        tf.greater(self.decoder_pos_input, tf.zeros_like(self.decoder_pos_input)), tf.float32)
        self.copy_gate_mask = tf.concat([self.copy_gate_mask, tf.zeros([tf.shape(self.encoder_input)[0], 1], tf.float32)], 1)

        # self.debug1 = tf.print(self.gpt_out_mask, [tf.shape(self.gpt_out_mask)], summarize=1000)


        # ### how to condition on encoder?
        # with tf.variable_scope(scope_name):
        #     st_1, st_2 = en_state
        #     cond_hidden0 = tf.concat([st_1, st_2], 1)
        #     self.cond_hidden0 = tf.layers.dense(cond_hidden0, self.hidden_size)
        #     self.cond_hidden0 = tf.expand_dims(self.cond_hidden0, 1)
        #     start_len = tf.shape(self.gpt_context)[1]
        #     self.cond_hidden0 = tf.tile(self.cond_hidden0, [1, start_len, 1])

        # # decoder for training
        # de_outputs, de_state, self.de_conv_loss, self.copy_gate = self.decoder_t(en_state, self.decoder_embed, self.decoder_len)
        # # decoder for testing
        # self.g_tokens, self.atts = self.decoder_g(en_state)


        # with tf.device("/gpu:1"):
        #     ### start with context tokens of domain
        #     ### start reuse of gpt
        #     # context_outputs = self.step_gpt(self.gpt_hparams, self.gpt_context, self.batch_size)
        #     context_outputs = self.step_gpt(self.gpt_hparams, self.encoder_input, self.batch_size)
        #     logits0 = context_outputs['logits'][:, -1, :]
        #     dist0 = tf.nn.softmax(logits0) # start token
        #     x0 = tf.cast(tf.argmax(dist0, 1), tf.int32)
        #     past0 = context_outputs['presents']
        #     hidden0 = context_outputs['hidden'][:, -1, :]


        # decoder for training
        de_outputs, _, self.de_conv_loss, self.copy_gate_loss = self.decoder_t(self.decoder_input, self.decoder_len, x0, past0, hidden0)
        # decoder for testing
        self.g_tokens, self.atts = self.decoder_g(x0, past0, hidden0)
        # self.beam_seqs, self.beam_probs, self.cand_seqs, self.cand_probs = self.decoder_beam(en_state, beam_size)
        

        # ======================================== losses ======================================== #



        ### enc-dec loss
        self.decoder_output_one_hot = tf.one_hot(indices=self.decoder_output, 
                                                depth=self.target_vocab,
                                                axis=-1)
        ### mask for dec. plus eos
        dec_shape_len = tf.shape(self.decoder_output)[1]
        batch_nums = tf.range(0, dec_shape_len)
        batch_nums = tf.expand_dims(batch_nums, 0)
        batch_nums = tf.tile(batch_nums, [self.batch_size, 1])
        decoder_len_com = tf.expand_dims(self.decoder_len, 1)
        decoder_len_com = tf.tile(decoder_len_com, [1, dec_shape_len])
        mask = tf.cast(
                tf.less_equal(batch_nums, decoder_len_com), tf.float32)


        losses = -tf.reduce_sum(self.decoder_output_one_hot * tf.log(de_outputs + 1e-6), 2)
        losses = mask * losses

        ### faster. original reduce mean
        self.mean_loss = tf.reduce_sum(losses)

        self.de_conv_loss *= self.coverage_penalty

        self.copy_gate_loss = self.copy_gate_penalty * tf.reduce_sum(self.copy_gate_loss)

        if self.use_copy_gate:
            self.mean_loss += self.copy_gate_loss

        if self.use_coverage:
            self.mean_loss += self.de_conv_loss



        # train_var = tf.trainable_variables()
        # train_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seq2seq')

        train_params = tf.trainable_variables()


        ### train enc-dec
        with tf.variable_scope(scope_name):

            # grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), self.grad_clip)
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, train_params, colocate_gradients_with_ops=True), self.grad_clip)
        #     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #     self.train_op = optimizer.apply_gradients(zip(self.grads, train_params), global_step=self.global_step)

        # self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)



        ### accumulate gradient
        # with tf.variable_scope(scope_name):
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

            self.acc_gradients = list(map(lambda param: tf.get_variable(param.name.split(":")[0],
                                                                    param.get_shape(), param.dtype,
                                                                    tf.constant_initializer(0.0), trainable=False),
                                                                    train_params))

            self._loss = tf.get_variable("acc_loss", (), tf.float32, tf.constant_initializer(0.0), trainable=False)
            self._cov_loss = tf.get_variable("acc_cov_loss", (), tf.float32, tf.constant_initializer(0.0), trainable=False)
            self._gate_loss = tf.get_variable("acc_gate_loss", (), tf.float32, tf.constant_initializer(0.0), trainable=False)

            # We abuse the gradient descent optimizer for accumulating gradients and loss (summing)
            acc_opt = tf.train.GradientDescentOptimizer(-1.0)
            self.accumulate_gradients = acc_opt.apply_gradients(zip(self.grads, self.acc_gradients))
            self.acc_loss = acc_opt.apply_gradients([(self.mean_loss, self._loss)])
            self.acc_cov_loss = acc_opt.apply_gradients([(self.de_conv_loss, self._cov_loss)])
            self.acc_gate_loss = acc_opt.apply_gradients([(self.copy_gate_loss, self._gate_loss)])


            # train update
            self.update = self.opt.apply_gradients(
                zip(list(map(lambda v: v.value(), self.acc_gradients)), train_params), global_step=self.global_step)

            self.reset = list(map(lambda param: param.initializer, self.acc_gradients))
            self.reset.append(self._loss.initializer)
            self.reset.append(self._cov_loss.initializer)
            self.reset.append(self._gate_loss.initializer)


        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)





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
        lm_output = model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        hidden = lm_output['hidden']
        presents.set_shape(past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits, # [batch, sequence, hparams.n_vocab]
            'presents': presents,
            'hidden': hidden
        }

    def decoder_t(self, inputs, inputs_len, x0, past0, hidden0):
        ### gather p_gen and att_weights
        batch_size = tf.shape(self.decoder_input)[0]
        max_time = tf.shape(self.decoder_input)[1]
        encoder_len = tf.shape(self.encoder_embed)[1]

        time = tf.constant(0, dtype=tf.int32)
        f0 = tf.zeros([batch_size], dtype=tf.bool)

        inputs_ta = tf.TensorArray(dtype=tf.int32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0]))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
        emit_gate = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        ### coverage mechanisim
        coverage_att_sum = tf.zeros([batch_size, encoder_len], dtype=tf.float32)
        covloss0 = 0.0

        # gpt_mask_layer = tf.expand_dims(self.gpt_out_mask, 0)
        # gpt_mask_layer = tf.tile(gpt_mask_layer, [batch_size, 1])


        def loop_fn(t, x_t, past, hidden, emit_ta, emit_gate, coverage_att_sum, covloss, finished):


            # with tf.device("/gpu:1"):
            ### gpt generate
            next_outputs = self.step_gpt(self.gpt_hparams, x_t[:, tf.newaxis], self.batch_size, past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(1) # temperator = 1
            o_dist = tf.nn.softmax(logits) 

            # o_dist *= self.gpt_out_mask

            past_nt = tf.concat([past, next_outputs['presents']], axis=-2)
            hidden_nt = next_outputs['hidden'][:, -1, :]


            # ### try with tune
            # logits = self.dec_out(hidden_nt, finished)
            # # logits = self.dec_out(hidden_nt_cond, finished)
            # o_dist = tf.nn.softmax(logits)
            # # o_dist = tf.multiply(o_dist, gpt_mask_layer)
            # # o_dist = tf.divide(o_dist, (1e-6 + tf.reduce_sum(o_dist, 1, keepdims=True)))


            ### concat field pos 
            batch_nums_time = tf.range(0, limit=batch_size)
            time_batch = tf.fill([batch_size], t)
            collect_ind = tf.stack([batch_nums_time, time_batch], axis=1)
            this_field_pos_emb = tf.gather_nd(self.decoder_field_pos_emb, collect_ind) # [batch_size, field + pos]
            att_x_in = tf.nn.embedding_lookup(self.embedding, x_t)
            att_x_in = tf.concat([att_x_in, this_field_pos_emb], axis=1)

            o_weight, p_gen = self.att_layer(hidden_nt, att_x_in, hidden, coverage_att_sum, self.en_outputs, self.field_pos_embed, finished=finished)


            ### o_weight = batch * len, already normalized. p_gen = batch * 1
            out_dist = p_gen * o_dist # batch * self.target_vocab
            # att_dist = (1 - p_gen) * o_weight # batch * len
            att_dist = o_weight


            batch_nums = tf.range(0, limit=batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)
            batch_nums = tf.tile(batch_nums, [1, encoder_len])
            indices = tf.stack((batch_nums, self.encoder_input), axis=2) # batch_size * enc_len * 2
            shape = [batch_size, self.target_vocab]
            attn_dists_projected = tf.scatter_nd(indices, att_dist, shape)

            # final_dists = out_dist + attn_dists_projected
            final_dists = out_dist + (1 - p_gen) * attn_dists_projected

            copy_mask = tf.gather_nd(self.copy_gate_mask, collect_ind)

            # final_dists = tf.where(tf.cast(copy_mask, tf.bool), attn_dists_projected, final_dists)

            ### for gate loss
            # emit_gate = emit_gate.write(t, tf.sigmoid(p_gen))
            emit_gate = emit_gate.write(t, tf.multiply(p_gen, copy_mask))
            emit_ta = emit_ta.write(t, final_dists)



            ### coverage
            this_covloss = tf.reduce_sum(tf.minimum(coverage_att_sum, o_weight))
            covloss += this_covloss
            coverage_att_sum += o_weight


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
        batch_size = tf.shape(self.encoder_input)[0]
        encoder_len = tf.shape(self.encoder_embed)[1]

        time = tf.constant(0, dtype=tf.int32)
        f0 = tf.zeros([batch_size], dtype=tf.bool)

        ### concat with field + pos + rpos to input
        x0_field = tf.nn.embedding_lookup(self.embedding, tf.fill([batch_size], self.empty_token))
        x0_pos = tf.nn.embedding_lookup(self.pembedding, tf.zeros([batch_size], dtype=tf.int32))
        x0_rpos = tf.nn.embedding_lookup(self.rembedding, tf.zeros([batch_size], dtype=tf.int32))
        field_pos0 = tf.concat([x0_field, x0_pos, x0_rpos], 1)

        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
        att_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        ### coverage mechanisim
        coverage_att_sum = tf.zeros([batch_size, encoder_len], dtype=tf.float32)

        att_mask = tf.ones([batch_size, encoder_len], dtype=tf.float32)

        # gpt_mask_layer = tf.expand_dims(self.gpt_out_mask, 0)
        # gpt_mask_layer = tf.tile(gpt_mask_layer, [batch_size, 1])

        def loop_fn(t, x_t, past, hidden, field_pos_emb, emit_ta, att_ta, coverage_att_sum, att_mask, finished):



            # with tf.device("/gpu:1"):
            ### gpt generate
            next_outputs = self.step_gpt(self.gpt_hparams, x_t[:, tf.newaxis], self.batch_size, past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(1) # temperator = 1
            o_dist = tf.nn.softmax(logits) 
            # o_dist *= self.gpt_out_mask

            past_nt = tf.concat([past, next_outputs['presents']], axis=-2)
            hidden_nt = next_outputs['hidden'][:, -1, :]


            # ### try with tune
            # logits = self.dec_out(hidden_nt, finished)
            # # logits = self.dec_out(hidden_nt_cond, finished)
            # o_dist = tf.nn.softmax(logits)
            # # o_dist = tf.multiply(o_dist, gpt_mask_layer)
            # # o_dist = tf.divide(o_dist, (1e-6 + tf.reduce_sum(o_dist, 1, keepdims=True)))


            att_x_in = tf.nn.embedding_lookup(self.embedding, x_t)
            att_x_in = tf.concat([att_x_in, field_pos_emb], axis=1)

            o_weight, p_gen = self.att_layer(hidden_nt, att_x_in, hidden, coverage_att_sum, self.en_outputs, self.field_pos_embed, finished=finished)


            ### o_weight = batch * len, already normalized. p_gen = batch * 1
            out_dist = p_gen * o_dist # batch * self.target_vocab
            att_dist = (1 - p_gen) * o_weight # batch * len

            ### mask previous
            att_dist *= att_mask


            batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
            batch_nums = tf.tile(batch_nums, [1, encoder_len]) # shape (batch_size, enc_len)
            indices = tf.stack((batch_nums, self.encoder_input), axis=2) # shape (batch_size, enc_len, 2)
            shape = [batch_size, self.target_vocab]
            attn_dists_projected = tf.scatter_nd(indices, att_dist, shape) # batch * target_vocab

            final_dists = out_dist + attn_dists_projected

            # final_dists = out_dist


            ### coverage
            coverage_att_sum += o_weight


            emit_ta = emit_ta.write(t, final_dists)
            att_ta = att_ta.write(t, tf.transpose(o_weight, [1,0]))
            x_nt = tf.cast(tf.argmax(final_dists, 1), tf.int32)



            ### field pos emb next round
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




            ### mask atten pos of previous
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



    def __call__(self, x, sess, mode):


        # ### normal training
        # loss,  _, copy_gate_loss, de_conv_loss = sess.run([self.mean_loss, self.train_op, self.copy_gate_loss, self.de_conv_loss],
        #                                                {self.encoder_input: x['enc_in'], self.encoder_len: x['enc_len'], 
        #                                                 self.encoder_field: x['enc_fd'], self.encoder_pos: x['enc_pos'], 
        #                                                 self.encoder_rpos: x['enc_rpos'], self.decoder_input: x['dec_in'],
        #                                                 self.decoder_len: x['dec_len'], self.decoder_output: x['dec_out'],
        #                                                 self.decoder_field_input: x['dec_field'], self.decoder_pos_input: x['dec_pos'],
        #                                                 self.decoder_rpos_input: x['dec_rpos'], self.gpt_context: x['gpt_context'],
        #                                                 self.context: x['context']})


        # return loss, copy_gate_loss, de_conv_loss

        ### accumulate gradient
        if mode == 0:
            loss, copy_gate_loss, de_conv_loss, _, _, _, _ = sess.run([self.mean_loss, self.copy_gate_loss, self.de_conv_loss, self.accumulate_gradients, self.acc_loss, \
                                                                    self.acc_cov_loss, self.acc_gate_loss],
                                                           {self.encoder_input: x['enc_in'], self.encoder_len: x['enc_len'], 
                                                            self.encoder_field: x['enc_fd'], self.encoder_pos: x['enc_pos'], 
                                                            self.encoder_rpos: x['enc_rpos'], self.decoder_input: x['dec_in'],
                                                            self.decoder_len: x['dec_len'], self.decoder_output: x['dec_out'],
                                                            self.decoder_field_input: x['dec_field'], self.decoder_pos_input: x['dec_pos'],
                                                            self.decoder_rpos_input: x['dec_rpos'], self.gpt_context: x['gpt_context'],
                                                            self.context: x['context']})


            return loss, copy_gate_loss, de_conv_loss, 0

        ### update
        if mode == 1:

            acc_loss, acc_cov_loss, acc_gate_loss = sess.run([self._loss, self._cov_loss, self._gate_loss])
            sess.run(self.update)
            sess.run(self.reset)


            return acc_loss, acc_gate_loss, acc_cov_loss

    def generate(self, x, sess):
        predictions, atts = sess.run([self.g_tokens, self.atts],
                               {self.encoder_input: x['enc_in'], self.encoder_field: x['enc_fd'], 
                                self.encoder_len: x['enc_len'], self.encoder_pos: x['enc_pos'],
                                self.encoder_rpos: x['enc_rpos'], self.decoder_input: x['dec_in'],
                                self.gpt_context: x['gpt_context'], self.context: x['context']})
        return predictions, atts

    def generate_beam(self, x, sess):
        # beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all
        beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all = sess.run(
                         [self.beam_seqs,self.beam_probs, self.cand_seqs, self.cand_probs],
                         {self.encoder_input: x['enc_in'], self.encoder_field: x['enc_fd'],
                          self.encoder_len: x['enc_len'], self.encoder_pos: x['enc_pos'],
                          self.encoder_rpos: x['enc_rpos']})
        return beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all

    def save(self, path, sess):

        checkpoint_path = os.path.join(path, "wiki2bio_model.ckpt")
        self.saver.save(sess, checkpoint_path, global_step=self.global_step.eval())
        print ("Model saved on global step %d." % (self.global_step.eval()))

    def load(self, path, sess):
        ckpt = tf.train.get_checkpoint_state(path)
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        g.saver.restore(sess, ckpt.model_checkpoint_path)


















