import tensorflow as tf
import numpy as np


class StegoNet(object):

    def alice_model(self, collection, img, msg):
        with tf.contrib.framework.arg_scope(
            [tf.contrib.layers.fully_connected, tf.contrib.layers.conv2d],
                variables_collections=[collection]):

            # Embed message into image
            img_stack = tf.concat(axis=3, values=[img, msg])
            conv0 = tf.contrib.layers.conv2d(
                img_stack, 4, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            conv1 = tf.contrib.layers.conv2d(
                conv0, 4, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            conv2 = tf.contrib.layers.conv2d(
                conv1, 3, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            return tf.identity(conv2, name=collection + '_out')

    def bob_model(self, collection, img):

        with tf.contrib.framework.arg_scope(
            [tf.contrib.layers.fully_connected, tf.contrib.layers.conv2d],
                variables_collections=[collection]):

            conv0 = tf.contrib.layers.conv2d(
                img, 4, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            conv1 = tf.contrib.layers.conv2d(
                conv0, 4, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            conv2 = tf.contrib.layers.conv2d(
                conv1, 1, 1, 1, 'SAME', activation_fn=tf.nn.tanh)

            return tf.identity(conv2, name=collection + '_out')

    def eve_model(self, collection, img):

        with tf.contrib.framework.arg_scope(
            [tf.contrib.layers.fully_connected, tf.contrib.layers.conv2d],
                variables_collections=[collection]):

            conv0 = tf.contrib.layers.conv2d(
                img, 4, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            conv1 = tf.contrib.layers.conv2d(
                conv0, 4, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            conv2 = tf.contrib.layers.conv2d(
                conv1, 1, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            flattened = tf.contrib.layers.flatten(conv2)

            eve_out = tf.contrib.layers.fully_connected(
                flattened, 1, activation_fn=tf.sigmoid)

            return tf.identity(eve_out, name=collection + '_out')

    def __init__(self, config):
        self.cfg = config
        img_batch = tf.placeholder(tf.float32, shape=(
            None, self.cfg.IMG_SIZE, self.cfg.IMG_SIZE, self.cfg.NUM_CHANNELS), name="img_in")
        msg_batch = tf.placeholder(tf.float32, shape=(
            None, self.cfg.IMG_SIZE, self.cfg.IMG_SIZE, 1), name="msg_in")

        alice_out = self.alice_model('alice', img_batch, msg_batch)
        with tf.variable_scope('bob_vars'):
            bob_out = self.bob_model('bob', alice_out)
        with tf.variable_scope('bob_vars', reuse=True):
            _ = self.bob_model('bob_eval', img_batch)

        # eve_gt_vals = tf.random_uniform(
        #    [self.cfg.BATCH_SIZE], 0, 2, dtype=tf.int32)
        #mask = tf.equal(eve_gt_vals, tf.constant(1))
        #eve_in = tf.where(mask, alice_out, img_batch)
        #eve_gt = tf.cast(eve_gt_vals, tf.float32)

        # with tf.variable_scope('eve_vars'):
        #    eve_out = self.eve_model('eve', eve_in)
        # with tf.variable_scope('eve_vars', reuse=True):
        #    _ = self.eve_model('eve_eval', img_batch)

        # self.reset_eve_vars = tf.group(
        #    *[w.initializer for w in tf.get_collection('eve')]
        # )

        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.cfg.LEARNING_RATE)

        # Eve loss
        # self.eve_loss = tf.reduce_sum(
        #    tf.abs(eve_out - eve_gt), name='eve_loss')
        # self.eve_optimizer = optimizer.minimize(
        #    self.eve_loss, var_list=tf.get_collection('eve'), name='eve_optimizer')

        # Alice & bob loss
        bob_img_diff = tf.reduce_sum(
            tf.square(img_batch - alice_out), [1, 2, 3])
        self.bob_img_loss = tf.reduce_sum(bob_img_diff, name='img_loss')
        self.bob_bits_wrong = tf.reduce_sum(
            tf.abs((bob_out + 1.) / 2. - (msg_batch + 1.) / 2.), [1])
        self.bob_reconstruction_loss = tf.reduce_sum(
            self.bob_bits_wrong, name='bob_reconstruction_loss')
        # bob_eve_error_deviation = tf.abs(
        #    float(self.cfg.BATCH_SIZE) / 2. - self.eve_loss)
        # bob_eve_loss = tf.reduce_sum(
        #    tf.square(bob_eve_error_deviation) / (self.cfg.BATCH_SIZE / 2)**2)
        self.bob_loss = (self.bob_reconstruction_loss /
                         self.cfg.IMG_SIZE**2 + self.bob_img_loss)
        self.bob_optimizer = optimizer.minimize(
            self.bob_reconstruction_loss, var_list=(tf.get_collection('alice'), tf.get_collection('bob')), name='bob_optimizer')
        self.alice_bob_optimizer = optimizer.minimize(
            self.bob_loss, var_list=(tf.get_collection('alice'), tf.get_collection('bob')), name='alice_bob_optimizer')
