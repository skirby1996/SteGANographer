import tensorflow as tf

from src.model import StegoNet


def main():
    with tf.Session as sess:
        stego_net = StegoNet(sess, msg_len=64, key_len=16,
                             batch_size=64, epochs=16, learning_rate=0.001)

        stego_net.train()


if __name__ == '__main__':
    main()
