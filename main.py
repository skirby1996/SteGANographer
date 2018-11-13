import sys
import os

import numpy as np
import tensorflow as tf

from PIL import Image
from src.config import Config
from src.dataset import Dataset
from src.model import StegoNet


def random_image_batch(batch_size, img_size, num_channels):
    '''
    Arguments:
        batch_size: Number of images to generate
        img_size: Pixel height/width of image
        num_channels: Channel depth of image
    Returns:
        batch: A [batch_size, img_size, img_size, num_channels]
               array of (0., 1.) floating values
    '''

    # Generates random pixel vals
    # batch = np.random.randint(0, 256, size=(
    #     batch_size, img_size, img_size, num_channels)).astype(float)
    # return batch / 255.

    batch = np.random.randint(2, size=(
        batch_size, img_size, img_size, num_channels)).astype(float)

    return ((batch * 2) - 1)


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


def eval(sess, cfg, dst, img_loss_op, bob_reconstruction_loss_op, n):
    '''
    Evaluates the current network on n batches of random examples

    Arguments:
        sess: The current tensorflow session
        cfg: An instance of the Config class
        dst: An instance of the Dataset class
        eve_loss_op: Eve's loss tensor
        bob_reconstruction_loss_op: Bob's reconstruction loss tensor
        n: The number of iterations to run

    Returns:
        bob_loss: Bob's error rate
        eve_loss: Eve's error rate
    '''

    ib = dst.get_batch(cfg.BATCH_SIZE, cfg.IMG_SIZE)
    mb = random_image_batch(cfg.BATCH_SIZE, cfg.IMG_SIZE, 1)

    bob_loss, img_loss = sess.run(
        [bob_reconstruction_loss_op, img_loss_op], feed_dict={'img_in:0': ib, 'msg_in:0': mb})

    return bob_loss / cfg.BATCH_SIZE, img_loss / cfg.BATCH_SIZE


def train(cfg, model_dir, train_set, val_set):
    # Create folders for model weights and logging
    save_dir = os.path.join(model_dir, "data")
    log_dir = os.path.join(model_dir, "logs")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # Create log file or resume previous
    current_epoch = 0
    log_file_name = os.path.join(log_dir, cfg.MODEL_NAME + "_log.csv")
    if os.path.isfile(log_file_name):
        with open(log_file_name, 'r') as log_file:
            lines = log_file.readlines()
            if len(lines) > 1:
                current_epoch = int(lines[-1].split(',')[0])
                print("Resuming from epoch %d..." % current_epoch)
                log_file = open(log_file_name, 'a')
    if current_epoch == 0:
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
            _ = StegoNet(cfg)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

        # Get initial loss from starting weights
        bob_loss, img_loss = eval(
            sess, cfg, val_set, 'img_loss:0', 'bob_reconstruction_loss:0', 16)
        if (current_epoch == 0):
            log_file.write("0,%f,%f\n" % (bob_loss, img_loss))

        print("Training for %d epochs..." % (cfg.NUM_EPOCHS))
        for epoch in range(current_epoch, current_epoch + cfg.NUM_EPOCHS):
            for _ in range(cfg.ITERS_PER_ACTOR * cfg.ALICE_MULTIPLIER):
                ib = train_set.get_batch(cfg.BATCH_SIZE, cfg.IMG_SIZE)
                mb = random_image_batch(cfg.BATCH_SIZE, cfg.IMG_SIZE, 1)
                # mb, kb = get_batch(cfg.BATCH_SIZE, cfg.MSG_SIZE, cfg.KEY_SIZE)
                sess.run('alice_bob_optimizer', feed_dict={
                         'img_in:0': ib, 'msg_in:0': mb})
            for _ in range(cfg.ITERS_PER_ACTOR * cfg.BOB_MULTIPLIER):
                ib = train_set.get_batch(cfg.BATCH_SIZE, cfg.IMG_SIZE)
                mb = random_image_batch(cfg.BATCH_SIZE, cfg.IMG_SIZE, 1)
                # mb, kb = get_batch(cfg.BATCH_SIZE, cfg.MSG_SIZE, cfg.KEY_SIZE)
                sess.run('bob_optimizer', feed_dict={
                         'img_in:0': ib, 'msg_in:0': mb})
            if (epoch + 1) % cfg.LOG_CHECKPOINT == 0:
                bob_loss, img_loss = eval(
                    sess, cfg, val_set, 'img_loss:0', 'bob_reconstruction_loss:0', 16)
                log_file.write("%d,%f,%f\n" % (epoch + 1, bob_loss, img_loss))

            # Display progress in console
            prog = int(20. * (epoch - current_epoch + 1)/(cfg.NUM_EPOCHS))
            prog_bar = "[%s%s%s]" % (
                '=' * prog, ('=' if prog == 20 else '>'), '.' * (20 - prog),)
            print("Epoch %06d/%06d - %s\tbob_loss: %f\timg_loss: %f" % (
                epoch + 1, current_epoch + cfg.NUM_EPOCHS, prog_bar, bob_loss, img_loss), end="\r", flush=True)
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

def production_test(cfg, model_dir, dst):
    save_dir = os.path.join(model_dir, "data")
    log_dir = os.path.join(model_dir, "logs")
    results_dir = os.path.join(model_dir, "results")

    if not os.path.exists(save_dir):
        print("Error: Model \"%s\" does not exist" % cfg.MODEL_NAME)
        return False

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

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

        num_tests = 5000
        bob_passed = 0
        for test in range(num_tests):
            #print("\nTest %d:" % (test + 1))
            ib = dst.get_batch(1, cfg.IMG_SIZE)
            mb = random_image_batch(1, cfg.IMG_SIZE, 1)
            msg_bin = nn_to_bin((mb + 1.) / 2.)

            image_norm = (ib * 255.).astype('uint8')[0]
            image = Image.fromarray(image_norm, 'RGB')
            # image.save(os.path.join(
            #    results_dir, (str(test) + "_original.bmp")))

            a_out = sess.run('alice_vars_1/alice_eval_out:0', feed_dict={
                'img_in:0': ib, 'msg_in:0': mb})

            alice_out_norm = (a_out * 255.).astype('uint8')[0]
            alice_out_img = Image.fromarray(alice_out_norm, 'RGB')
            # alice_out_img.save(os.path.join(
            #    results_dir, (str(test) + "_embedded.bmp")))

            a_out *= 255.
            a_out = a_out.astype(int)
            a_out = a_out.astype(float)
            a_out /= 255.

            b_out = sess.run('bob_vars_1/bob_eval_out:0',
                             feed_dict={'img_in:0': a_out})
            #bob_out_bin = nn_to_bin((b_out + 1.) / 2.)
            bob_out_bin = nn_to_bin(b_out)
            #print("bob_out: ", bob_out_bin)

            bob_missed_bits = sess.run(
                tf.reduce_sum(tf.abs(msg_bin - bob_out_bin)))
            if bob_missed_bits == 0:
                bob_passed += 1
            else:
                image.save(os.path.join(
                    results_dir, (str(test) + "_original.bmp")))
                alice_out_img.save(os.path.join(
                    results_dir, (str(test) + "_embedded.bmp")))

            #print("Bob missed: ", bob_missed_bits)
            # Display progress in console
            prog = int(20. * (test + 1)/(num_tests))
            prog_bar = "[%s%s%s]" % (
                '=' * prog, ('=' if prog == 20 else '>'), '.' * (20 - prog),)
            print("Test %06d/%06d - %s\tFailed: %d" % (
                test + 1, num_tests, prog_bar, test - bob_passed + 1), end="\r", flush=True)

        print("\nFinal Results:")
        print("Bob recovered: [%d/%d]" % (bob_passed, num_tests))


def main():
    cfg = Config()
    cfg.print_summary()

    train_set = Dataset(cfg, "train")
    val_set = Dataset(cfg, "val")

    # Parse command line args

    # Make dirs for model saving and logging
    root_dir = os.path.abspath("")
    model_dir = os.path.join(root_dir, "models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_dir = os.path.join(model_dir, cfg.MODEL_NAME)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    train(cfg, model_dir, train_set, val_set)
    production_test(cfg, model_dir, val_set)


if __name__ == '__main__':
    main()
