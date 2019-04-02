#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-5-12 下午10:47
# @Author  : Tianyu Liu

import tensorflow as tf
import pickle


class dualAttentionWrapper(object):
    def __init__(self, emb_size, hidden_size, input_size, field_size, scope_name):
        ### here input_size == hidden_size
        # self.hs = tf.transpose(hs, [1,0,2])  # input_len * batch * input_size
        # self.fds = tf.transpose(fds, [1,0,2])
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.field_size = field_size
        self.scope_name = scope_name
        self.params = {}

        self.emb_size = emb_size

        with tf.variable_scope(scope_name):
            self.Wh = tf.get_variable('Wh', [input_size, hidden_size])
            # self.Wh = tf.get_variable('Wh', [emb_size, hidden_size])
            self.bh = tf.get_variable('bh', [hidden_size])
            self.Ws = tf.get_variable('Ws', [input_size, hidden_size])
            self.bs = tf.get_variable('bs', [hidden_size])
            self.Wo = tf.get_variable('Wo', [2*input_size, hidden_size])
            self.bo = tf.get_variable('bo', [hidden_size])
            self.Wf = tf.get_variable('Wf', [field_size, hidden_size])
            self.bf = tf.get_variable('bf', [hidden_size])
            self.Wr = tf.get_variable('Wr', [input_size, hidden_size])
            self.br = tf.get_variable('br', [hidden_size])

            ### coverage
            self.Wc = tf.get_variable('Wc', [1])
            self.bc = tf.get_variable('bc', [1])

            ### add pointer params
            ### p_gen = sigmod(wh * ht + ws * st + wx * xt + bptr)
            self.wh_ptr = tf.get_variable('wh_ptr', [self.hidden_size, 1])
            # self.wh_ptr = tf.get_variable('wh_ptr', [self.emb_size, 1])
            self.ws_ptr = tf.get_variable('ws_ptr', [self.hidden_size, 1])
            self.wx_ptr = tf.get_variable('wx_ptr', [self.emb_size, 1])
            self.b_ptr = tf.get_variable('b_ptr', [1])

        self.params.update({'Wh': self.Wh, 'Ws': self.Ws, 'Wo': self.Wo,
                            'bh': self.bh, 'bs': self.bs, 'bo': self.bo,
                            'Wf': self.Wf, 'Wr': self.Wr, 
                            'bf': self.bf, 'br': self.br,
                            'wh_ptr': self.wh_ptr, 'ws_ptr': self.ws_ptr,
                            'wx_ptr': self.wx_ptr, 'b_ptr': self.b_ptr,
                            'Wc': self.Wc, 'bc': self.bc})

        # hs2d = tf.reshape(self.hs, [-1, input_size])
        # phi_hs2d = tf.tanh(tf.nn.xw_plus_b(hs2d, self.Wh, self.bh))
        # self.phi_hs = tf.reshape(phi_hs2d, tf.shape(self.hs))
        # fds2d = tf.reshape(self.fds, [-1, field_size])
        # phi_fds2d = tf.tanh(tf.nn.xw_plus_b(fds2d, self.Wf, self.bf))
        # self.phi_fds = tf.reshape(phi_fds2d, tf.shape(self.hs))

    def __call__(self, x, in_t, last_x, coverage_att_sum, hs, fds, finished = None):

        hs = tf.transpose(hs, [1,0,2])  # input_len * batch * input_size
        fds = tf.transpose(fds, [1,0,2])

        hidden_shape = [tf.shape(hs)[0], tf.shape(hs)[1], self.hidden_size]

        hs2d = tf.reshape(hs, [-1, self.input_size])
        phi_hs2d = tf.tanh(tf.nn.xw_plus_b(hs2d, self.Wh, self.bh))
        phi_hs = tf.reshape(phi_hs2d, tf.shape(hs))
        fds2d = tf.reshape(fds, [-1, self.field_size])
        phi_fds2d = tf.tanh(tf.nn.xw_plus_b(fds2d, self.Wf, self.bf))
        phi_fds = tf.reshape(phi_fds2d, tf.shape(hs))

        # hs2d = tf.reshape(hs, [-1, self.emb_size])
        # phi_hs2d = tf.tanh(tf.nn.xw_plus_b(hs2d, self.Wh, self.bh))
        # phi_hs = tf.reshape(phi_hs2d, hidden_shape)
        # fds2d = tf.reshape(fds, [-1, self.field_size])
        # phi_fds2d = tf.tanh(tf.nn.xw_plus_b(fds2d, self.Wf, self.bf))
        # phi_fds = tf.reshape(phi_fds2d, hidden_shape)

        ### add coverage: coverage_att_sum # batch * enc_len
        ### how to incorporate coverage penalty? for each or for all?
        gamma_h = tf.tanh(tf.nn.xw_plus_b(x, self.Ws, self.bs))  # batch * hidden_size
        alpha_h = tf.tanh(tf.nn.xw_plus_b(x, self.Wr, self.br))
        fd_weights = tf.reduce_sum(phi_fds * alpha_h, reduction_indices=2, keepdims=True) # len * batch * 1
        fd_weights = tf.exp(fd_weights - tf.reduce_max(fd_weights, reduction_indices=0, keepdims=True))
        fd_weights = tf.divide(fd_weights, (1e-6 + tf.reduce_sum(fd_weights, reduction_indices=0, keepdims=True))) # len * batch * 1


        weights = tf.reduce_sum(phi_hs * gamma_h, reduction_indices=2, keepdims=True)  # input_len * batch * 1


        weights = tf.exp(weights - tf.reduce_max(weights, reduction_indices=0, keepdims=True))
        weights = tf.divide(weights, (1e-6 + tf.reduce_sum(weights, reduction_indices=0, keepdims=True)))
        weights = tf.divide(weights * fd_weights, (1e-6 + tf.reduce_sum(weights * fd_weights, reduction_indices=0, keepdims=True))) # len * batch * 1

        ### coverage
        # coverage_penalty = tf.tanh(tf.nn.xw_plus_b(coverage_att_sum, self.Wc, self.bc))
        coverage_penalty = tf.tanh(coverage_att_sum * self.Wc + self.bc)
        coverage_penalty = tf.expand_dims(tf.transpose(coverage_penalty, [1,0]), -1) # enc_len * batch * 1
        coverage_penalty = tf.exp(coverage_penalty - tf.reduce_max(coverage_penalty, reduction_indices=0, keepdims=True))
        weights = tf.divide(weights * coverage_penalty, (1e-6 + tf.reduce_sum(weights * coverage_penalty, reduction_indices=0, keepdims=True))) # len * batch * 1



        
        context = tf.reduce_sum(hs * weights, reduction_indices=0)  # batch * input_size
        # out = tf.tanh(tf.nn.xw_plus_b(tf.concat([context, x], -1), self.Wo, self.bo))

        #### pointer generator
        ### p_gen = sigmod(wh * ht + ws * st + wx * xt + bptr)
        p_gen = tf.matmul(context, self.wh_ptr) + tf.matmul(last_x, self. ws_ptr) + tf.matmul(in_t, self.wx_ptr) + self.b_ptr
        p_gen = tf.sigmoid(p_gen) # batch * 1

        weights = tf.squeeze(weights, 2) # len * batch
        weights = tf.transpose(weights, [1,0]) # batch * len

        if finished is not None:
            # out = tf.where(finished, tf.zeros_like(out), out)
            p_gen = tf.where(finished, tf.ones_like(p_gen), p_gen)
            weights = tf.where(finished, tf.zeros_like(weights), weights)

        return weights, p_gen

    def save(self, path):
        param_values = {}
        for param in self.params:
            param_values[param] = self.params[param].eval()
        with open(path, 'wb') as f:
            pickle.dump(param_values, f, True)

    def load(self, path):
        param_values = pickle.load(open(path, 'rb'))
        for param in param_values:
            self.params[param].load(param_values[param])
