#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:37
# @Author  : Tianyu Liu

import tensorflow as tf
import pickle
from AttentionUnit import AttentionWrapper
from dualAttentionUnit import dualAttentionWrapper
from LstmUnit import LstmUnit
from fgateLstmUnit import fgateLstmUnit
from OutputUnit import OutputUnit


class SeqUnit(object):
    def __init__(self, batch_size, hidden_size, emb_size, field_size, pos_size, source_vocab, field_vocab,
                 position_vocab, target_vocab, field_concat, position_concat, fgate_enc, dual_att,
                 encoder_add_pos, decoder_add_pos, learning_rate, scope_name, name, start_token=2, stop_token=2, max_length=150):
        '''
        batch_size, hidden_size, emb_size, field_size, pos_size: size of batch; hidden layer; word/field/position embedding
        source_vocab, target_vocab, field_vocab, position_vocab: vocabulary size of encoder words; decoder words; field types; position
        field_concat, position_concat: bool values, whether concat field/position embedding to word embedding for encoder inputs or not
        fgate_enc, dual_att: bool values, whether use field-gating / dual attention or not
        encoder_add_pos, decoder_add_pos: bool values, whether add position embedding to field-gating encoder / decoder with dual attention or not
        '''
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.field_size = field_size
        self.pos_size = pos_size
        self.uni_size = emb_size if not field_concat else emb_size+field_size
        self.uni_size = self.uni_size if not position_concat else self.uni_size+2*pos_size
        self.field_encoder_size = field_size if not encoder_add_pos else field_size+2*pos_size
        self.field_attention_size = field_size if not decoder_add_pos else field_size+2*pos_size
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.field_vocab = field_vocab
        self.position_vocab = position_vocab
        self.grad_clip = 5.0
        self.start_token = start_token
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

        self.units = {}
        self.params = {}

        self.encoder_input = tf.placeholder(tf.int32, [None, None])
        self.encoder_field = tf.placeholder(tf.int32, [None, None])
        self.encoder_pos = tf.placeholder(tf.int32, [None, None])
        self.encoder_rpos = tf.placeholder(tf.int32, [None, None])
        self.decoder_input = tf.placeholder(tf.int32, [None, None])
        self.encoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_output = tf.placeholder(tf.int32, [None, None])
        self.enc_mask = tf.sign(tf.to_float(self.encoder_pos))
        with tf.variable_scope(scope_name):
            if self.fgate_enc:
                print 'field-gated encoder LSTM'
                self.enc_lstm = fgateLstmUnit(self.hidden_size, self.uni_size, self.field_encoder_size, 'encoder_select')
            else:
                print 'normal encoder LSTM'
                self.enc_lstm = LstmUnit(self.hidden_size, self.uni_size, 'encoder_lstm')
            self.dec_lstm = LstmUnit(self.hidden_size, self.emb_size, 'decoder_lstm')
            self.dec_out = OutputUnit(self.hidden_size, self.target_vocab, 'decoder_output')

        self.units.update({'encoder_lstm': self.enc_lstm,'decoder_lstm': self.dec_lstm,
                           'decoder_output': self.dec_out})

        # ======================================== embeddings ======================================== #
        #with tf.device('/cpu:0'):
        with tf.variable_scope(scope_name):
            self.embedding = tf.get_variable('embedding', [self.source_vocab, self.emb_size])
            self.encoder_embed = tf.nn.embedding_lookup(self.embedding, self.encoder_input)
            self.decoder_embed = tf.nn.embedding_lookup(self.embedding, self.decoder_input)
            if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos:
                self.fembedding = tf.get_variable('fembedding', [self.field_vocab, self.field_size])
                self.field_embed = tf.nn.embedding_lookup(self.fembedding, self.encoder_field)
                self.field_pos_embed = self.field_embed
                if self.field_concat:
                    self.encoder_embed = tf.concat([self.encoder_embed, self.field_embed], 2)
            if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
                self.pembedding = tf.get_variable('pembedding', [self.position_vocab, self.pos_size])
                self.rembedding = tf.get_variable('rembedding', [self.position_vocab, self.pos_size])
                self.pos_embed = tf.nn.embedding_lookup(self.pembedding, self.encoder_pos)
                self.rpos_embed = tf.nn.embedding_lookup(self.rembedding, self.encoder_rpos)
                if position_concat:
                    self.encoder_embed = tf.concat([self.encoder_embed, self.pos_embed, self.rpos_embed], 2)
                    self.field_pos_embed = tf.concat([self.field_embed, self.pos_embed, self.rpos_embed], 2)
                elif self.encoder_add_pos or self.decoder_add_pos:
                    self.field_pos_embed = tf.concat([self.field_embed, self.pos_embed, self.rpos_embed], 2)

        if self.field_concat or self.fgate_enc:
            self.params.update({'fembedding': self.fembedding})
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            self.params.update({'pembedding': self.pembedding})
            self.params.update({'rembedding': self.rembedding})
        self.params.update({'embedding': self.embedding})

        # ======================================== encoder ======================================== #
        if self.fgate_enc:
            print 'field gated encoder used'
            en_outputs, en_state = self.fgate_encoder(self.encoder_embed, self.field_pos_embed, self.encoder_len)
        else:
            print 'normal encoder used'
            en_outputs, en_state = self.encoder(self.encoder_embed, self.encoder_len)
        # ======================================== decoder ======================================== #

        if self.dual_att:
	        print 'dual attention mechanism used'
	        with tf.variable_scope(scope_name):
	            self.att_layer = dualAttentionWrapper(self.emb_size, self.hidden_size, self.hidden_size, self.field_attention_size,
	                                                    en_outputs, self.field_pos_embed, "attention")
	            self.units.update({'attention': self.att_layer})
        else:
            print "normal attention used"
            with tf.variable_scope(scope_name):
                self.att_layer = AttentionWrapper(self.hidden_size, self.hidden_size, en_outputs, "attention")
                self.units.update({'attention': self.att_layer})


        # decoder for training
        de_outputs, de_state = self.decoder_t(en_state, self.decoder_embed, self.decoder_len)
        # decoder for testing
        self.g_tokens, self.atts = self.decoder_g(en_state)
        # self.beam_seqs, self.beam_probs, self.cand_seqs, self.cand_probs = self.decoder_beam(en_state, beam_size)
        

        self.decoder_output_one_hot = tf.one_hot(indices=self.decoder_output, 
                                                depth=self.target_vocab,
                                                axis=-1)

        ### original loss with logits
        #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=de_outputs, labels=self.decoder_output)

        losses = -tf.reduce_sum(self.decoder_output_one_hot * tf.log(de_outputs + 1e-9), 2)


        mask = tf.sign(tf.to_float(self.decoder_output))
        losses = mask * losses
        ### faster. original reduce mean
        self.mean_loss = tf.reduce_sum(losses)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def encoder(self, inputs, inputs_len):
        batch_size = tf.shape(self.encoder_input)[0]
        max_time = tf.shape(self.encoder_input)[1]
        hidden_size = self.hidden_size

        time = tf.constant(0, dtype=tf.int32)
        h0 = (tf.zeros([batch_size, hidden_size], dtype=tf.float32),
              tf.zeros([batch_size, hidden_size], dtype=tf.float32))
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0,2]))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, s_t, emit_ta, finished):
            o_t, s_nt = self.enc_lstm(x_t, s_t, finished)
            emit_ta = emit_ta.write(t, o_t)
            finished = tf.greater_equal(t+1, inputs_len)
            x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.uni_size], dtype=tf.float32),
                                     lambda: inputs_ta.read(t+1))
            return t+1, x_nt, s_nt, emit_ta, finished

        _, _, state, emit_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, inputs_ta.read(0), h0, emit_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        return outputs, state

    def fgate_encoder(self, inputs, fields, inputs_len):
        batch_size = tf.shape(self.encoder_input)[0]
        max_time = tf.shape(self.encoder_input)[1]
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


    def decoder_t(self, initial_state, inputs, inputs_len):
        ### gather p_gen and att_weights
        batch_size = tf.shape(self.decoder_input)[0]
        max_time = tf.shape(self.decoder_input)[1]
        encoder_len = tf.shape(self.encoder_input)[1]

        time = tf.constant(0, dtype=tf.int32)
        h0 = initial_state
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        x0 = tf.nn.embedding_lookup(self.embedding, tf.fill([batch_size], self.start_token))
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0,2]))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        # emit_p_gen = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, s_t, emit_ta, finished):
            o_t, s_nt = self.dec_lstm(x_t, s_t, finished)
            o_t, o_weight, p_gen = self.att_layer(o_t, x_t, s_t)
            o_t = self.dec_out(o_t, finished) # batch * self.target_vocab

            ### pointer generator
            #emit_ta = emit_ta.write(t, o_t)

            ### o_weight = batch * len, already normalized. p_gen = batch * 1
            out_dist = p_gen * tf.nn.softmax(o_t) # batch * self.target_vocab
            att_dist = (1 - p_gen) * o_weight # batch * len

            batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
            batch_nums = tf.tile(batch_nums, [1, encoder_len]) # shape (batch_size, enc_len)
            indices = tf.stack((batch_nums, self.encoder_input), axis=2) # shape (batch_size, enc_len, 2)
            shape = [batch_size, self.target_vocab]
            attn_dists_projected = tf.scatter_nd(indices, att_dist, shape)

            final_dists = out_dist + attn_dists_projected
            emit_ta = emit_ta.write(t, final_dists)


            finished = tf.greater_equal(t, inputs_len)
            x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.emb_size], dtype=tf.float32),
                                     lambda: inputs_ta.read(t))
            return t+1, x_nt, s_nt, emit_ta, finished

        _, _, state, emit_ta,  _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, x0, h0, emit_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        return outputs, state

    def decoder_g(self, initial_state):
        batch_size = tf.shape(self.encoder_input)[0]
        encoder_len = tf.shape(self.encoder_input)[1]

        time = tf.constant(0, dtype=tf.int32)
        h0 = initial_state
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        x0 = tf.nn.embedding_lookup(self.embedding, tf.fill([batch_size], self.start_token))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
        att_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, s_t, emit_ta, att_ta, finished):
            o_t, s_nt = self.dec_lstm(x_t, s_t, finished)
            o_t, o_weight, p_gen = self.att_layer(o_t, x_t, s_t)
            o_t = self.dec_out(o_t, finished)

            ### pointer generator
            #emit_ta = emit_ta.write(t, o_t)

            ### o_weight = batch * len, already normalized. p_gen = batch * 1
            out_dist = p_gen * tf.nn.softmax(o_t) # batch * self.target_vocab
            att_dist = (1 - p_gen) * o_weight # batch * len

            batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
            batch_nums = tf.tile(batch_nums, [1, encoder_len]) # shape (batch_size, enc_len)
            indices = tf.stack((batch_nums, self.encoder_input), axis=2) # shape (batch_size, enc_len, 2)
            shape = [batch_size, self.target_vocab]
            attn_dists_projected = tf.scatter_nd(indices, att_dist, shape) # batch * target_vocab

            final_dists = out_dist + attn_dists_projected


            emit_ta = emit_ta.write(t, final_dists)
            att_ta = att_ta.write(t, tf.transpose(o_weight, [1,0]))
            next_token = tf.arg_max(final_dists, 1)
            x_nt = tf.nn.embedding_lookup(self.embedding, next_token)
            finished = tf.logical_or(finished, tf.equal(next_token, self.stop_token))
            finished = tf.logical_or(finished, tf.greater_equal(t, self.max_length))
            return t+1, x_nt, s_nt, emit_ta, att_ta, finished

        _, _, state, emit_ta, att_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, _5, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, x0, h0, emit_ta, att_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        pred_tokens = tf.arg_max(outputs, 2)
        atts = att_ta.stack()
        return pred_tokens, atts


    def decoder_beam(self, initial_state, beam_size):

        def beam_init():
            # return beam_seqs_1 beam_probs_1 cand_seqs_1 cand_prob_1 next_states time
            time_1 = tf.constant(1, dtype=tf.int32)
            beam_seqs_0 = tf.constant([[self.start_token]]*beam_size)
            beam_probs_0 = tf.constant([0.]*beam_size)

            cand_seqs_0 = tf.constant([[self.start_token]])
            cand_probs_0 = tf.constant([-3e38])

            beam_seqs_0._shape = tf.TensorShape((None, None))
            beam_probs_0._shape = tf.TensorShape((None,))
            cand_seqs_0._shape = tf.TensorShape((None, None))
            cand_probs_0._shape = tf.TensorShape((None,))
            
            inputs = [self.start_token]
            x_t = tf.nn.embedding_lookup(self.embedding, inputs)
            print(x_t.get_shape().as_list())
            o_t, s_nt = self.dec_lstm(x_t, initial_state)
            o_t, w_t = self.att_layer(o_t)
            o_t = self.dec_out(o_t)
            print(s_nt[0].get_shape().as_list())
            # initial_state = tf.reshape(initial_state, [1,-1])
            logprobs2d = tf.nn.log_softmax(o_t)
            total_probs = logprobs2d + tf.reshape(beam_probs_0, [-1, 1])
            total_probs_noEOS = tf.concat([tf.slice(total_probs, [0, 0], [1, self.stop_token]),
                               tf.tile([[-3e38]], [1, 1]),
                               tf.slice(total_probs, [0, self.stop_token + 1],
                                        [1, self.target_vocab - self.stop_token - 1])], 1)
            flat_total_probs = tf.reshape(total_probs_noEOS, [-1])
            print flat_total_probs.get_shape().as_list()

            beam_k = tf.minimum(tf.size(flat_total_probs), beam_size)
            next_beam_probs, top_indices = tf.nn.top_k(flat_total_probs, k=beam_k)

            next_bases = tf.floordiv(top_indices, self.target_vocab)
            next_mods = tf.mod(top_indices, self.target_vocab)

            next_beam_seqs = tf.concat([tf.gather(beam_seqs_0, next_bases),
                                        tf.reshape(next_mods, [-1, 1])], 1)

            cand_seqs_pad = tf.pad(cand_seqs_0, [[0, 0], [0, 1]])
            beam_seqs_EOS = tf.pad(beam_seqs_0, [[0, 0], [0, 1]])
            new_cand_seqs = tf.concat([cand_seqs_pad, beam_seqs_EOS], 0)
            print new_cand_seqs.get_shape().as_list()

            EOS_probs = tf.slice(total_probs, [0, self.stop_token], [beam_size, 1])
            new_cand_probs = tf.concat([cand_probs_0, tf.reshape(EOS_probs, [-1])], 0)
            cand_k = tf.minimum(tf.size(new_cand_probs), self.beam_size)
            next_cand_probs, next_cand_indices = tf.nn.top_k(new_cand_probs, k=cand_k)
            next_cand_seqs = tf.gather(new_cand_seqs, next_cand_indices)

            part_state_0 = tf.reshape(tf.stack([s_nt[0]]*beam_size), [beam_size, self.hidden_size])
            part_state_1 = tf.reshape(tf.stack([s_nt[1]]*beam_size), [beam_size, self.hidden_size])
            part_state_0._shape = tf.TensorShape((None, None))
            part_state_1._shape = tf.TensorShape((None, None))
            next_states = (part_state_0, part_state_1)
            print next_states[0].get_shape().as_list()
            return next_beam_seqs, next_beam_probs, next_cand_seqs, next_cand_probs, next_states, time_1

        beam_seqs_1, beam_probs_1, cand_seqs_1, cand_probs_1, states_1, time_1 = beam_init()
        beam_seqs_1._shape = tf.TensorShape((None, None))
        beam_probs_1._shape = tf.TensorShape((None,))
        cand_seqs_1._shape = tf.TensorShape((None, None))
        cand_probs_1._shape = tf.TensorShape((None,))
        # states_1._shape = tf.TensorShape((2, None, self.hidden_size))
        def beam_step(beam_seqs, beam_probs, cand_seqs, cand_probs, states, time):
            '''
            beam_seqs : [beam_size, time]
            beam_probs: [beam_size, ]
            cand_seqs : [beam_size, time]
            cand_probs: [beam_size, ]
            states : [beam_size * hidden_size, beam_size * hidden_size]
            '''
            inputs = tf.reshape(tf.slice(beam_seqs, [0, time], [beam_size, 1]), [beam_size])
            # print inputs.get_shape().as_list()
            x_t = tf.nn.embedding_lookup(self.embedding, inputs)
            # print(x_t.get_shape().as_list())
            o_t, s_nt = self.dec_lstm(x_t, states)
            o_t, w_t = self.att_layer(o_t)
            o_t = self.dec_out(o_t)
            logprobs2d = tf.nn.log_softmax(o_t)
            print logprobs2d.get_shape().as_list()
            total_probs = logprobs2d + tf.reshape(beam_probs, [-1, 1])
            print total_probs.get_shape().as_list()
            total_probs_noEOS = tf.concat([tf.slice(total_probs, [0, 0], [beam_size, self.stop_token]),
                                           tf.tile([[-3e38]], [beam_size, 1]),
                                           tf.slice(total_probs, [0, self.stop_token + 1],
                                                    [beam_size, self.target_vocab - self.stop_token - 1])], 1)
            print total_probs_noEOS.get_shape().as_list()
            flat_total_probs = tf.reshape(total_probs_noEOS, [-1])
            print flat_total_probs.get_shape().as_list()

            beam_k = tf.minimum(tf.size(flat_total_probs), beam_size)
            next_beam_probs, top_indices = tf.nn.top_k(flat_total_probs, k=beam_k)
            print next_beam_probs.get_shape().as_list()

            next_bases = tf.floordiv(top_indices, self.target_vocab)
            next_mods = tf.mod(top_indices, self.target_vocab)
            print next_mods.get_shape().as_list()

            next_beam_seqs = tf.concat([tf.gather(beam_seqs, next_bases),
                                        tf.reshape(next_mods, [-1, 1])], 1)
            next_states = (tf.gather(s_nt[0], next_bases), tf.gather(s_nt[1], next_bases))
            print next_beam_seqs.get_shape().as_list()

            cand_seqs_pad = tf.pad(cand_seqs, [[0, 0], [0, 1]])
            beam_seqs_EOS = tf.pad(beam_seqs, [[0, 0], [0, 1]])
            new_cand_seqs = tf.concat([cand_seqs_pad, beam_seqs_EOS], 0) 
            print new_cand_seqs.get_shape().as_list()

            EOS_probs = tf.slice(total_probs, [0, self.stop_token], [beam_size, 1])
            new_cand_probs = tf.concat([cand_probs, tf.reshape(EOS_probs, [-1])], 0)
            cand_k = tf.minimum(tf.size(new_cand_probs), self.beam_size)
            next_cand_probs, next_cand_indices = tf.nn.top_k(new_cand_probs, k=cand_k)
            next_cand_seqs = tf.gather(new_cand_seqs, next_cand_indices)

            return next_beam_seqs, next_beam_probs, next_cand_seqs, next_cand_probs, next_states, time+1
        
        def beam_cond(beam_probs, beam_seqs, cand_probs, cand_seqs, state, time):
            length =  (tf.reduce_max(beam_probs) >= tf.reduce_min(cand_probs))
            return tf.logical_and(length, tf.less(time, 60) )
            # return tf.less(time, 18)

        loop_vars = [beam_seqs_1, beam_probs_1, cand_seqs_1, cand_probs_1, states_1, time_1]
        ret_vars = tf.while_loop(cond=beam_cond, body=beam_step, loop_vars=loop_vars, back_prop=False)
        beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all, _, time_all = ret_vars

        return beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all

    def __call__(self, x, sess):
        loss,  _ = sess.run([self.mean_loss, self.train_op],
                           {self.encoder_input: x['enc_in'], self.encoder_len: x['enc_len'], 
                            self.encoder_field: x['enc_fd'], self.encoder_pos: x['enc_pos'], 
                            self.encoder_rpos: x['enc_rpos'], self.decoder_input: x['dec_in'],
                            self.decoder_len: x['dec_len'], self.decoder_output: x['dec_out']})
        return loss

    def generate(self, x, sess):
        predictions, atts = sess.run([self.g_tokens, self.atts],
                               {self.encoder_input: x['enc_in'], self.encoder_field: x['enc_fd'], 
                                self.encoder_len: x['enc_len'], self.encoder_pos: x['enc_pos'],
                                self.encoder_rpos: x['enc_rpos']})
        return predictions, atts

    def generate_beam(self, x, sess):
        # beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all
        beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all = sess.run(
                         [self.beam_seqs,self.beam_probs, self.cand_seqs, self.cand_probs],
                         {self.encoder_input: x['enc_in'], self.encoder_field: x['enc_fd'],
                          self.encoder_len: x['enc_len'], self.encoder_pos: x['enc_pos'],
                          self.encoder_rpos: x['enc_rpos']})
        return beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all

    def save(self, path):
        for u in self.units:
            self.units[u].save(path+u+".pkl")
        param_values = {}
        for param in self.params:
            param_values[param] = self.params[param].eval()
        with open(path+self.name+".pkl", 'wb') as f:
            pickle.dump(param_values, f, True)

    def load(self, path):
        for u in self.units:
            self.units[u].load(path+u+".pkl")
        param_values = pickle.load(open(path+self.name+".pkl", 'rb'))
        for param in param_values:
            self.params[param].load(param_values[param])
