import sys
import os

import numpy as np
import tensorflow as tf

from PIL import Image
from src.config import Config


def nn_to_bin(batch):
    return np.around(batch).astype(int)


def load_model(cfg, model_dir):
    save_dir = os.path.join(model_dir, "data")
    log_dir = os.path.join(model_dir, "logs")

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

        return sess

        num_tests = 50
        bob_passed = 0
        for test in range(num_tests):
            print("\nTest %d:" % (test + 1))
            ib = dst.get_batch(1, cfg.IMG_SIZE)
            mb = random_image_batch(1, cfg.IMG_SIZE, 1)
            msg_bin = nn_to_bin((mb + 1.) / 2.)

            image_norm = (ib * 255.).astype('uint8')[0]
            image = Image.fromarray(image_norm, 'RGB')
            image.save(os.path.join(
                results_dir, (str(test) + "_original.bmp")))

            a_out = sess.run('alice_out:0', feed_dict={
                'img_in:0': ib, 'msg_in:0': mb})

            alice_out_norm = (a_out * 255.).astype('uint8')[0]
            alice_out_img = Image.fromarray(alice_out_norm, 'RGB')
            alice_out_img.save(os.path.join(
                results_dir, (str(test) + "_embedded.bmp")))

            b_out = sess.run('bob_vars_1/bob_eval_out:0',
                             feed_dict={'img_in:0': a_out})
            bob_out_bin = nn_to_bin((b_out + 1.) / 2.)
            #print("bob_out: ", bob_out_bin)

            bob_missed_bits = sess.run(
                tf.reduce_sum(tf.abs(msg_bin - bob_out_bin)))
            if bob_missed_bits == 0:
                bob_passed += 1
            print("Bob missed: ", bob_missed_bits)

        print("\nFinal Results:")
        print("Bob recovered: [%d/%d]" % (bob_passed, num_tests))


def main():
    cfg = Config()

    # Make dirs for model saving and logging
    root_dir = os.path.abspath("")
    model_dir = os.path.join(root_dir, "models")
    model_dir = os.path.join(model_dir, cfg.MODEL_NAME)
    if not os.path.exists(model_dir):
        print("Error: No model exists at %s" % model_dir)
        return False
    sess = load_model(cfg, model_dir)

    sess.close()


if __name__ == '__main__':
    main()
