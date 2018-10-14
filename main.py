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
            for j in range(cfg.ITERS_PER_ACTOR):
                sess.run(stego_net.bob_optimizer)
            for j in range(cfg.ITERS_PER_ACTOR * cfg.EVE_MULTIPLIER):
                sess.run(stego_net.eve_optimizer)
            if (i + 1) % cfg.LOG_CHECKPOINT == 0:
                bob_loss, eve_loss = eval(sess, cfg, stego_net, 16)
                print("Epoch %d/%d:" % (i + 1, cfg.NUM_EPOCHS))
                print("bob_loss: %.6f\neve_loss: %.6f\n" %
                      (bob_loss, eve_loss))


def main():
    cfg = Config()
    train(cfg)


if __name__ == '__main__':
    main()
