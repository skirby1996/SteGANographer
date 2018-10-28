import sys
import os

import numpy as np
import tensorflow as tf

from src.config import Config
from src.model import StegoNet


def random_boolean_batch(batch_size, n):
    '''
    Arguments:
        batch_size: Number of arrays to return
        n: Length in random boolean batch
    Returns:
        batch: A [batch_size, n] array of (-1./1.) boolean numbers
    '''

    batch = np.random.randint(2, size=(batch_size, n))
    batch = (batch * 2) - 1
    return batch.astype(float)


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


def nn_to_bin(batch):
    return np.around(batch).astype(int)


def main():
    cfg = Config()
    cfg.print_summary()

    # Parse command line args

    # Make dirs for model saving and logging
    root_dir = os.path.abspath("")
    model_dir = os.path.join(root_dir, "models")
    save_dir = os.path.join(model_dir, "data")
    log_dir = os.path.join(model_dir, "logs")

    if not os.path.exists(save_dir):
        print("Error: Model \"%s\" does not exist" % cfg.MODEL_NAME)
        return False

    # Use log file to infer current epoch
    current_epoch = 0
    log_file_name = os.path.join(log_dir, cfg.MODEL_NAME + "_log.csv")
    if os.path.isfile(log_file_name):
        with open(log_file_name, 'r') as log_file:
            lines = log_file.readlines()
            if len(lines) > 1:
                current_epoch = int(lines[-1].split(',')[0])
                print("Current epoch: %d\n" % current_epoch)
            else:
                print("Error: Cannot infer epoch from log file...")
                return False
    else:
        print("Error: No log file from which to infer epoch")
        return False

    # Restore saved weights
    weight_file_name = cfg.MODEL_NAME + '_train-' + str(current_epoch)
    meta_file_name = weight_file_name + '.meta'

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            os.path.join(save_dir, meta_file_name))
        saver.restore(sess, tf.train.latest_checkpoint(save_dir))

        replay = True
        while replay:
            msg = input("Enter a message: ")
            key = input("Enter a key: ")

            # Convert msg and key to binary
            # Pad msg and key as needed
            # Build message and key batch
            mb = None
            kb = None

            a_out = sess.run('alice_out:0', feed_dict={
                'msg_in:0': mb, 'key_in:0': kb})

            b_out = sess.run('bob_vars_1/bob_eval_out:0',
                             feed_dict={'msg_in:0': a_out, 'key_in:0': kb})
            bob_out_bin = nn_to_bin((b_out + 1.) / 2.)

            e_out = sess.run('eve_vars_1/eve_eval_out:0',
                             feed_dict={'msg_in:0': a_out})
            eve_out_bin = nn_to_bin((e_out + 1.) / 2.)

            # Convert binary to back to string
            bob_out = ""
            eve_out = ""

            print("bob_out: ", bob_out)
            print("eve_out: ", eve_out)

            user_in = input("Encrypt another message? (y/n)")
            if userin


if __name__ == '__main__':
    main()
