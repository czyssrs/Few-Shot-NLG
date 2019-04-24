import tensorflow as tf
import pickle


class mlpUnit(object):
    def __init__(self, hidden_size, input_size, scope_name):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.scope_name = scope_name
        self.params = {}

        with tf.variable_scope(scope_name):
            self.W = tf.get_variable('W', [self.input_size, self.hidden_size])
            self.b = tf.get_variable('b', [self.hidden_size], initializer=tf.zeros_initializer([self.hidden_size]), dtype=tf.float32)

        self.params.update({'W':self.W, 'b':self.b})

    def __call__(self, x, s, finished = None):
        h_prev, c_prev = s # dummy, of no use here

        c = tf.nn.xw_plus_b(x, self.W, self.b)

        # Final Memory cell
        h = tf.relu(c)

        out, state = h, (h, c) # state is dummy
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