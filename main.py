import sys
import os

import tensorflow as tf

from src.config import Config
from src.model import StegoNet


def eval(sess, cfg, eve_loss_op, bob_reconstruction_loss_op, n):
    '''
    Evaluates the current network on n batches of random examples

    Arguments:
        sess: The current tensorflow session
        cfg: An instance of the Config class
        eve_loss_op: Eve's loss tensor
        bob_reconstruction_loss_op: Bob's reconstruction loss tensor
        n: The number of iterations to run

    Returns:
        bob_loss: Bob's error rate
        eve_loss: Eve's error rate
    '''
    bob_loss_total = 0
    eve_loss_total = 0
    for _ in range(n):
        bl, el = sess.run(
            [bob_reconstruction_loss_op, eve_loss_op])
        bob_loss_total += bl
        eve_loss_total += el
    bob_loss = bob_loss_total / (n * cfg.BATCH_SIZE)
    eve_loss = eve_loss_total / (n * cfg.BATCH_SIZE)
    return bob_loss, eve_loss


def train(cfg, model_dir):
    # Create folders for model weights and logging
    save_dir = os.path.join(model_dir, "data")
    log_dir = os.path.join(model_dir, "logs")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # Create log file or resume previous
    log_file_name = os.path.join(log_dir, cfg.MODEL_NAME + "_log.csv")
    if os.path.isfile(log_file_name):
        with open(log_file_name, 'r') as log_file:
            current_epoch = int(log_file.readlines()[-1].split(',')[0])
        print("Resuming from epoch %d..." % current_epoch)
        log_file = open(log_file_name, 'a')
    else:
        current_epoch = 0
        log_file = open(os.path.join(
            log_dir, cfg.MODEL_NAME + "_log.csv"), 'w')
        log_file.write("epoch,bob_loss,eve_loss\n")

    # Restore meta graph if loading previous model
    if current_epoch != 0:
        weight_file_name = cfg.MODEL_NAME + '_train-' + str(current_epoch)
        meta_file_name = weight_file_name + '.meta'

    with tf.Session() as sess:
        if current_epoch != 0:
            saver = tf.train.import_meta_graph(
                os.path.join(save_dir, meta_file_name))
            saver.restore(sess, tf.train.latest_checkpoint(save_dir))

        else:
            stego_net = StegoNet(cfg)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

        # Get initial loss from starting weights
        bob_loss, eve_loss = eval(
            sess, cfg, 'eve_loss:0', 'bob_reconstruction_loss:0', 16)
        if (current_epoch == 0):
            log_file.write("0,%f,%f\n" % (bob_loss, eve_loss))

        print("Training for %d epochs..." % (cfg.NUM_EPOCHS))
        for epoch in range(current_epoch, current_epoch + cfg.NUM_EPOCHS):
            for _ in range(cfg.ITERS_PER_ACTOR):
                sess.run('bob_optimizer')
            for _ in range(cfg.ITERS_PER_ACTOR * cfg.EVE_MULTIPLIER):
                sess.run('eve_optimizer')
            if (epoch + 1) % cfg.LOG_CHECKPOINT == 0:
                bob_loss, eve_loss = eval(
                    sess, cfg, 'eve_loss:0', 'bob_reconstruction_loss:0', 16)
                log_file.write("%d,%f,%f\n" % (epoch + 1, bob_loss, eve_loss))

            # Display progress in console
            prog = int(20. * (epoch - current_epoch + 1)/(cfg.NUM_EPOCHS))
            prog_bar = "[%s%s%s]" % (
                '=' * prog, ('=' if prog == 20 else '>'), '.' * (20 - prog),)
            print("Epoch %06d/%06d - %s\tbob_loss: %f\teve_loss: %f" % (
                epoch + 1, current_epoch + cfg.NUM_EPOCHS, prog_bar, bob_loss, eve_loss), end="\r", flush=True)
        print('\n')
        save_name = cfg.MODEL_NAME + "_train"
        saver.save(sess, os.path.join(save_dir, save_name),
                   global_step=(current_epoch + cfg.NUM_EPOCHS))
    log_file.close()


# def run_tests(cfg):
#     stego_net = StegoNet(cfg)
#     init = tf.global_variables_initializer()

#     # Test saving and loading models
#     with tf.Session() as sess:
#         sess.run(init)
#         bob_loss_orig, eve_loss_orig = eval(sess, cfg, stego_net, 16)


def main():
    cfg = Config()
    cfg.print_summary()

    # Parse command line args

    # Make dirs for model saving and logging
    root_dir = os.path.abspath("")
    model_dir = os.path.join(root_dir, "models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_dir = os.path.join(model_dir, cfg.MODEL_NAME)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    train(cfg, model_dir)


if __name__ == '__main__':
    main()
