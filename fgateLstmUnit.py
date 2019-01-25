#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-5-9 上午10:16
# @Author  : Tianyu Liu

import tensorflow as tf
import pickle


class fgateLstmUnit(object):
    def __init__(self, hidden_size, input_size, field_size, scope_name):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.field_size = field_size
        self.scope_name = scope_name
        self.params = {}

        with tf.variable_scope(scope_name):
            self.W = tf.get_variable('W', [self.input_size+self.hidden_size, 4*self.hidden_size])
            self.b = tf.get_variable('b', [4*self.hidden_size], initializer=tf.zeros_initializer(), dtype=tf.float32)
            self.W1 = tf.get_variable('W1', [self.field_size, 2*self.hidden_size])
            self.b1 = tf.get_variable('b1', [2*hidden_size], initializer=tf.zeros_initializer(), dtype=tf.float32)
        self.params.update({'W':self.W, 'b':self.b, 'W1':self.W1, 'b1':self.b1})

    def __call__(self, x, fd, s, finished = None):
        """
        :param x: batch * input
        :param s: (h,s,d)
        :param finished:
        :return:
        """
        h_prev, c_prev = s  # batch * hidden_size

        x = tf.concat([x, h_prev], 1)
        # fd = tf.concat([fd, h_prev], 1)
        i, j, f, o = tf.split(tf.nn.xw_plus_b(x, self.W, self.b), 4, 1)
        r, d = tf.split(tf.nn.xw_plus_b(fd, self.W1, self.b1), 2, 1)
        # Final Memory cell
        c = tf.sigmoid(f+1.0) * c_prev + tf.sigmoid(i) * tf.tanh(j) + tf.sigmoid(r) * tf.tanh(d)  # batch * hidden_size
        h = tf.sigmoid(o) * tf.tanh(c)

        out, state = h, (h, c)
        if finished is not None:
            out = tf.where(finished, tf.zeros_like(h), h)
            state = (tf.where(finished, h_prev, h), tf.where(finished, c_prev, c))
            # out = tf.multiply(1 - finished, h)
            # state = (tf.multiply(1 - finished, h) + tf.multiply(finished, h_prev),
            #          tf.multiply(1 - finished, c) + tf.multiply(finished, c_prev))

        return out, state

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