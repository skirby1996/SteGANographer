import tensorflow as tf
import numpy as np


class StegoNet(object):

    def alice_model(self, collection, img, msg, key):
        with tf.contrib.framework.arg_scope(
            [tf.contrib.layers.fully_connected, tf.contrib.layers.conv2d],
                variables_collections=[collection]):

            # Generate msg channel
            msg_fc0 = tf.contrib.layers.fully_connected(
                msg, self.cfg.MSG_SIZE,
                biases_initializer=tf.constant_initializer(0.),
                activation_fn=None)

            # msg_fc1 = tf.contrib.layers.fully_connected(
            #     msg_fc0, self.cfg.IMG_SIZE ** 2, activation_fn=tf.nn.tanh)

            msg_fc2 = tf.contrib.layers.fully_connected(
                msg_fc0, self.cfg.IMG_SIZE ** 2, activation_fn=tf.nn.sigmoid)

            msg_channel = tf.manip.reshape(
                msg_fc2, [tf.shape(img)[0], self.cfg.IMG_SIZE, self.cfg.IMG_SIZE, 1])

            # Generate key channel
            key_fc0 = tf.contrib.layers.fully_connected(
                key, self.cfg.MSG_SIZE,
                biases_initializer=tf.constant_initializer(0.),
                activation_fn=None)

            # key_fc1 = tf.contrib.layers.fully_connected(
            #     key_fc0, self.cfg.IMG_SIZE ** 2, activation_fn=tf.nn.tanh)

            key_fc2 = tf.contrib.layers.fully_connected(
                key_fc0, self.cfg.IMG_SIZE ** 2, activation_fn=tf.nn.sigmoid)

            key_channel = tf.manip.reshape(
                key_fc2, [tf.shape(img)[0], self.cfg.IMG_SIZE, self.cfg.IMG_SIZE, 1])

            # Convolve msg and key channels into embed channel
            stack = tf.concat(
                axis=3, values=[msg_channel, key_channel])
            embed_channel = tf.contrib.layers.conv2d(
                stack, 1, 1, 1)

            # Embed embed channel into image
            img_stack = tf.concat(axis=3, values=[img, embed_channel])
            # conv0 = tf.contrib.layers.conv2d(
            #     img_stack, 8, 2, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            # conv1 = tf.contrib.layers.conv2d(
            #     conv0, 8, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            # conv2 = tf.contrib.layers.conv2d(
            #     conv1, 8, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            conv3 = tf.contrib.layers.conv2d(
                img_stack, 3, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            return tf.to_float(conv3, name=collection + '_out')

    def bob_model(self, collection, img, key):

        with tf.contrib.framework.arg_scope(
            [tf.contrib.layers.fully_connected, tf.contrib.layers.conv2d],
                variables_collections=[collection]):

            # Attempt to extract channel from image
            # conv0 = tf.contrib.layers.conv2d(
            #     img, 8, 2, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            # conv1 = tf.contrib.layers.conv2d(
            #     conv0, 8, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            # conv2 = tf.contrib.layers.conv2d(
            #     conv1, 8, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            # conv3 = tf.contrib.layers.conv2d(
            #     conv2, 3, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            conv4 = tf.contrib.layers.conv2d(
                img, 1, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            # Generate key channel
            key_fc0 = tf.contrib.layers.fully_connected(
                key, self.cfg.MSG_SIZE,
                biases_initializer=tf.constant_initializer(0.),
                activation_fn=None)

            # key_fc1 = tf.contrib.layers.fully_connected(
            #     key_fc0, self.cfg.IMG_SIZE ** 2, activation_fn=tf.nn.tanh)

            key_fc2 = tf.contrib.layers.fully_connected(
                key_fc0, self.cfg.IMG_SIZE ** 2, activation_fn=tf.nn.sigmoid)

            key_channel = tf.manip.reshape(
                key_fc2, shape=[tf.shape(img)[0], self.cfg.IMG_SIZE, self.cfg.IMG_SIZE, 1])

            # Attempt to recover msg channel
            stack = tf.concat(axis=3, values=[conv4, key_channel])
            msg_channel = tf.contrib.layers.conv2d(
                stack, 1, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)
            msg_flattened = tf.contrib.layers.flatten(msg_channel)
            # Attempt to recover msg
            # fc0 = tf.contrib.layers.fully_connected(
            #     msg_channel, self.cfg.IMG_SIZE ** 2,
            #     activation_fn=tf.nn.sigmoid)

            fc1 = tf.contrib.layers.fully_connected(
                msg_flattened, self.cfg.MSG_SIZE ** 2, activation_fn=tf.nn.sigmoid)

            fc2 = tf.contrib.layers.fully_connected(
                fc1, self.cfg.MSG_SIZE, activation_fn=tf.nn.tanh)

            fc3 = tf.contrib.layers.fully_connected(
                fc2, self.cfg.MSG_SIZE, activation_fn=None)

            return tf.to_float(fc3, name=collection + "_out")

    def eve_model(self, collection, img):

        with tf.contrib.framework.arg_scope(
            [tf.contrib.layers.fully_connected, tf.contrib.layers.conv2d],
                variables_collections=[collection]):

            conv0 = tf.contrib.layers.conv2d(
                img, 2, 2, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            conv1 = tf.contrib.layers.conv2d(
                conv0, 2, 2, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            conv2 = tf.contrib.layers.conv2d(
                conv1, 1, 2, 1, 'SAME', activation_fn=tf.nn.sigmoid)

            flattened = tf.contrib.layers.flatten(conv2)
            # fc0 = tf.contrib.layers.fully_connected(
            #     flattened, 128, activation_fn=tf.sigmoid)

            fc1 = tf.contrib.layers.fully_connected(
                flattened, 32, activation_fn=tf.sigmoid)

            eve_out = tf.contrib.layers.fully_connected(
                fc1, 1, activation_fn=tf.sigmoid)

            return tf.to_float(eve_out, name=collection + '_out')

    def __init__(self, config):
        self.cfg = config
        img_batch = tf.placeholder(tf.float32, shape=(
            None, self.cfg.IMG_SIZE, self.cfg.IMG_SIZE, self.cfg.NUM_CHANNELS), name="img_in")
        msg_batch = tf.placeholder(tf.float32, shape=(
            None, self.cfg.MSG_SIZE), name="msg_in")
        key_batch = tf.placeholder(tf.float32, shape=(
            None, self.cfg.KEY_SIZE), name="key_in")

        alice_out = self.alice_model('alice', img_batch, msg_batch, key_batch)
        with tf.variable_scope('bob_vars'):
            bob_out = self.bob_model('bob', alice_out, key_batch)
        with tf.variable_scope('bob_vars', reuse=True):
            _ = self.bob_model('bob_eval', img_batch, key_batch)

        eve_gt_vals = tf.random_uniform(
            [self.cfg.BATCH_SIZE], 0, 2, dtype=tf.int32)
        mask = tf.equal(eve_gt_vals, tf.constant(1))
        eve_in = tf.where(mask, alice_out, img_batch)
        eve_gt = tf.cast(eve_gt_vals, tf.float32)

        with tf.variable_scope('eve_vars'):
            eve_out = self.eve_model('eve', eve_in)
        with tf.variable_scope('eve_vars', reuse=True):
            _ = self.eve_model('eve_eval', img_batch)

        self.reset_eve_vars = tf.group(
            *[w.initializer for w in tf.get_collection('eve')]
        )

        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.cfg.LEARNING_RATE)

        # Eve loss
        self.eve_loss = tf.reduce_sum(
            tf.abs(eve_out - eve_gt), name='eve_loss')
        self.eve_optimizer = optimizer.minimize(
            self.eve_loss, var_list=tf.get_collection('eve'), name='eve_optimizer')

        # Alice & bob loss
        self.bob_bits_wrong = tf.reduce_sum(
            tf.abs((bob_out + 1.) / 2. - (msg_batch + 1.) / 2.), [1])
        self.bob_reconstruction_loss = tf.reduce_sum(
            self.bob_bits_wrong, name='bob_reconstruction_loss')
        bob_eve_error_deviation = tf.abs(
            float(self.cfg.BATCH_SIZE) / 2. - self.eve_loss)
        bob_eve_loss = tf.reduce_sum(
            tf.square(bob_eve_error_deviation) / (self.cfg.BATCH_SIZE / 2)**2)
        self.bob_loss = (self.bob_reconstruction_loss /
                         self.cfg.MSG_SIZE + bob_eve_loss)
        self.bob_optimizer = optimizer.minimize(
            self.bob_loss, var_list=(tf.get_collection('alice'), tf.get_collection('bob')), name='bob_optimizer')
