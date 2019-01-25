#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:36
# @Author  : Tianyu Liu

import tensorflow as tf
import pickle


class OutputUnit(object):
    def __init__(self, input_size, output_size, scope_name):
        self.input_size = input_size
        self.output_size = output_size
        self.scope_name = scope_name
        self.params = {}

        with tf.variable_scope(scope_name):
            self.W = tf.get_variable('W', [input_size, output_size])
            self.b = tf.get_variable('b', [output_size], initializer=tf.zeros_initializer(), dtype=tf.float32)

        self.params.update({'W': self.W, 'b': self.b})

    def __call__(self, x, finished = None):
        out = tf.nn.xw_plus_b(x, self.W, self.b)

        if finished is not None:
            out = tf.where(finished, tf.zeros_like(out), out)
            #out = tf.multiply(1 - finished, out)
        return out

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
