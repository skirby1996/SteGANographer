import sys

import tensorflow as tf

from src.config import Config
from src.model import StegoNet

def eval(sess, cfg, stego_net, n):
    '''
    Evaluates the current network on n batches of random examples

    Arguments:
        sess: The current tensorflow session
        cfg: An instance of the Config class
        stego_net: An instance of the StegoNet class
        n: The number of iterations to trun

    Returns:
        bob_loss: Bob's error rate
        eve_loss: Eve's error rate
    '''
    bob_loss_total = 0
    eve_loss_total = 0
    for i in range(n):
        bl, el = sess.run(
            [stego_net.bob_reconstruction_loss, stego_net.eve_loss])
        bob_loss_total += bl
        eve_loss_total += el
    bob_loss = bob_loss_total / (n * cfg.BATCH_SIZE)
    eve_loss = eve_loss_total / (n * cfg.BATCH_SIZE)
    return bob_loss, eve_loss

def train(cfg):
    stego_net = StegoNet(cfg)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print("Training for %d epochs..." % (cfg.NUM_EPOCHS))
        for i in range(cfg.NUM_EPOCHS):
            if i == 0:
                # Get initial loss from starting weights
                bob_loss, eve_loss = eval(sess, cfg, stego_net, 16)
            for j in range(cfg.ITERS_PER_ACTOR):
                sess.run(stego_net.bob_optimizer)
            for j in range(cfg.ITERS_PER_ACTOR * cfg.EVE_MULTIPLIER):
                sess.run(stego_net.eve_optimizer)
            if (i + 1) % cfg.LOG_CHECKPOINT == 0:
                bob_loss, eve_loss = eval(sess, cfg, stego_net, 16)  
            prog = int(20. * (i + 1)/(cfg.NUM_EPOCHS))
            print("Epoch %06d/%06d - [%s>%s]\tbob_loss: %f\teve_loss: %f" % (i+1, cfg.NUM_EPOCHS,'='*prog, '.'*(20 - prog), bob_loss, eve_loss), end="\r", flush=True)
        save(cfg, sess)

def save(cfg, sess):
    train_saver = tf.train.Saver()
    train_saver.save(sess, "logs/my_model")

def run_tests(cfg):
    stego_net = StegoNet(cfg)
    init = tf.global_variables_initializer()

    # Test saving and loading models
    with tf.Session() as sess:
        sess.run(init)
        bob_loss_orig, eve_loss_orig = eval(sess, cfg, stego_net, 16)


def main():
    # Parse command line args
    
    # Make dir for model saving and logging

    cfg = Config()
    #cfg.print_summary()
    train(cfg)

if __name__ == '__main__':
    main()
