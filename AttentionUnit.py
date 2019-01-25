#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:35
# @Author  : Tianyu Liu

import tensorflow as tf
import pickle


class AttentionWrapper(object):
    def __init__(self, hidden_size, input_size, hs, scope_name):
        self.hs = tf.transpose(hs, [1,0,2])
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.scope_name = scope_name
        self.params = {}

        with tf.variable_scope(scope_name):
            self.Wh = tf.get_variable('Wh', [input_size, hidden_size])
            self.bh = tf.get_variable('bh', [hidden_size])
            self.Ws = tf.get_variable('Ws', [input_size, hidden_size])
            self.bs = tf.get_variable('bs', [hidden_size])
            self.Wo = tf.get_variable('Wo', [2*input_size, hidden_size])
            self.bo = tf.get_variable('bo', [hidden_size])
        self.params.update({'Wh': self.Wh, 'Ws': self.Ws, 'Wo': self.Wo,
                            'bh': self.bh, 'bs': self.bs, 'bo': self.bo})

        hs2d = tf.reshape(self.hs, [-1, input_size])
        phi_hs2d = tf.tanh(tf.nn.xw_plus_b(hs2d, self.Wh, self.bh))
        self.phi_hs = tf.reshape(phi_hs2d, tf.shape(self.hs))

    def __call__(self, x, finished = None):
        gamma_h = tf.tanh(tf.nn.xw_plus_b(x, self.Ws, self.bs))
        weights = tf.reduce_sum(self.phi_hs * gamma_h, reduction_indices=2, keep_dims=True)
        weight = weights
        weights = tf.exp(weights - tf.reduce_max(weights, reduction_indices=0, keep_dims=True))
        weights = tf.divide(weights, (1e-6 + tf.reduce_sum(weights, reduction_indices=0, keep_dims=True)))
        context = tf.reduce_sum(self.hs * weights, reduction_indices=0)
        # print wrt.get_shape().as_list()
        out = tf.tanh(tf.nn.xw_plus_b(tf.concat([context, x], -1), self.Wo, self.bo))

        if finished is not None:
            out = tf.where(finished, tf.zeros_like(out), out)
        return out, weights

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