import tensorflow as tf
import numpy as np
import keras


def random_boolean_batch(batch_size, n):
    '''
    Arguments:
        batch_size: Number of arrays to return
        n: Length in random boolean batch
    Returns:
        batch: A [batch_size, n] array of (-1./1.) boolean numbers
    '''

    batch = tf.random_uniform(
        [batch_size, n], minval=0, maxval=2, dtype=tf.int32)
    batch = (batch * 2) - 1
    return tf.cast(batch, tf.float32)


def get_batch(batch_size, msg_size, key_size):
    '''
    Arguments:
        batch_size: Number of messages and keys to generate
        msg_size: Bit length of each message
        key_size: Bit length of each key
    Returns:
        msg_batch: A [batch_size, msg_size] array of (-1./1.) boolean values
        key_batch: A [batch_size, key_size] array of (-1./1.) boolean values
    '''
    msg_batch = random_boolean_batch(batch_size, msg_size)
    key_batch = random_boolean_batch(batch_size, key_size)

    return msg_batch, key_batch


class StegoNet(object):
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

            return tf.squeeze(conv2, 2)

    def __init__(self, config):
        self.cfg = config

        msg_batch, key_batch = get_batch(
            self.cfg.BATCH_SIZE, self.cfg.MSG_SIZE, self.cfg.KEY_SIZE)
        alice_out = self.model('alice', msg_batch, key_batch)
        bob_out = self.model('bob', alice_out, key_batch)
        eve_out = self.model('eve', alice_out, None)

        self.reset_eve_vars = tf.group(
            *[w.initializer for w in tf.get_collection('eve')]
        )

        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.cfg.LEARNING_RATE)

        # Eve loss
        eve_bits_wrong = tf.reduce_sum(
            tf.abs((eve_out + 1.) / 2. - (msg_batch + 1.) / 2.), [1])
        self.eve_loss = tf.reduce_sum(eve_bits_wrong)
        self.eve_optimizer = optimizer.minimize(
            self.eve_loss, var_list=tf.get_collection('eve'))

        # Alice & bob loss
        self.bob_bits_wrong = tf.reduce_sum(
            tf.abs((bob_out + 1.) / 2. - (msg_batch + 1.) / 2.), [1])
        self.bob_reconstruction_loss = tf.reduce_sum(self.bob_bits_wrong)
        bob_eve_error_deviation = tf.abs(
            float(self.cfg.MSG_SIZE) / 2. - eve_bits_wrong)
        bob_eve_loss = tf.reduce_sum(
            tf.square(bob_eve_error_deviation) / (self.cfg.MSG_SIZE / 2)**2)
        self.bob_loss = (self.bob_reconstruction_loss /
                         self.cfg.MSG_SIZE + bob_eve_loss)

        self.bob_optimizer = optimizer.minimize(
            self.bob_loss, var_list=(tf.get_collection('alice'), tf.get_collection('bob')))
