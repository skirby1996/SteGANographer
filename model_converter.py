import sys
import os
import tensorflow as tf

from src.config import Config


def main():
    cfg = Config()

    # Prepare directories
    root_dir = os.path.abspath("")
    model_dir = os.path.join(root_dir, "models")
    model_dir = os.path.join(model_dir, cfg.MODEL_NAME)
    if not os.path.exists(model_dir):
        print("Error: No model exists at %s" % model_dir)
        return False

    target_dir = os.path.join(root_dir, "production")
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    out_dir = os.path.join(target_dir, "trained_models")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Load model
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

        scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'alice_vars') + \
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'bob_vars')

        saver = tf.train.Saver(scope)
        saver.save(sess, os.path.join(out_dir, 'StegoNet.cpkt'))


if __name__ == '__main__':
    main()
