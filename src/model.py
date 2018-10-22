import tensorflow as tf
import numpy as np


class StegoNet(object):
    '''
    The class containing the training model.

    TODO:
    Implement model saving/loading
    Convert training model into production model
    Move from a cipher encryption to a stegonographical embedder
    '''

    def model(self, collection, msg, key=None):
        if key is not None:
            msg_concat = tf.concat(axis=1, values=[msg, key])
        else:
            msg_concat = msg

        with tf.contrib.framework.arg_scope(
            [tf.contrib.layers.fully_connected, tf.contrib.layers.conv2d],
                variables_collections=[collection]): 

                fc0 = tf.contrib.layers.fully_connected(
                    msg_concat,
                    self.cfg.MSG_SIZE + self.cfg.KEY_SIZE,
                    biases_initializer=tf.constant_initializer(0.),
                    activation_fn=None)

                fc0 = tf.expand_dims(fc0, 2)

                conv0 = tf.contrib.layers.conv2d(
                    fc0, 2, 2, 2, 'SAME', activation_fn=tf.nn.sigmoid)

                conv1 = tf.contrib.layers.conv2d(
                    conv0, 2, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)

                conv2 = tf.contrib.layers.conv2d(
                    conv1, 1, 1, 1, 'SAME', activation_fn=tf.nn.tanh)

                return tf.squeeze(conv2, 2, name=collection + '_out')

    def __init__(self, config):
        self.cfg = config
        msg_batch = tf.placeholder(tf.float32, shape=(
            None, self.cfg.MSG_SIZE), name="msg_in")
        key_batch = tf.placeholder(tf.float32, shape=(
            None, self.cfg.KEY_SIZE), name="key_in")

        alice_out = self.model('alice', msg_batch, key_batch)        
        with tf.variable_scope('bob_vars'):
            bob_out = self.model('bob', alice_out, key_batch) 
        with tf.variable_scope('bob_vars', reuse=True):
            bob_eval_out = self.model('bob_eval', msg_batch, key_batch)
        eve_out = self.model('eve', alice_out, None)

        self.reset_eve_vars = tf.group(
            *[w.initializer for w in tf.get_collection('eve')]
        )

        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.cfg.LEARNING_RATE)

        # Eve loss
        eve_bits_wrong = tf.reduce_sum(
            tf.abs((eve_out + 1.) / 2. - (msg_batch + 1.) / 2.), [1])
        self.eve_loss = tf.reduce_sum(eve_bits_wrong, name='eve_loss')
        self.eve_optimizer = optimizer.minimize(
            self.eve_loss, var_list=tf.get_collection('eve'), name='eve_optimizer')

        # Alice & bob loss
        self.bob_bits_wrong = tf.reduce_sum(
            tf.abs((bob_out + 1.) / 2. - (msg_batch + 1.) / 2.), [1])
        self.bob_reconstruction_loss = tf.reduce_sum(
            self.bob_bits_wrong, name='bob_reconstruction_loss')
        bob_eve_error_deviation = tf.abs(
            float(self.cfg.MSG_SIZE) / 2. - eve_bits_wrong)
        bob_eve_loss = tf.reduce_sum(
            tf.square(bob_eve_error_deviation) / (self.cfg.MSG_SIZE / 2)**2)
        self.bob_loss = (self.bob_reconstruction_loss /
                         self.cfg.MSG_SIZE + bob_eve_loss)

        self.bob_optimizer = optimizer.minimize(
            self.bob_loss, var_list=(tf.get_collection('alice'), tf.get_collection('bob')), name='bob_optimizer')
