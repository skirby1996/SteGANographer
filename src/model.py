import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np


class StegoNet(object):
    def __init__(self, sess, msg_len=MSG_LEN, key_len=KEY_LEN,
                 batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
                 learning_rate=LEARNING_RATE):

        # Initiate parameters
        self.sess = sess
        self.msg_len = msg_len
        self.key_len = key_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    # Prepare data

    # Define models
    def build_model(self):
        self.alice = Sequential()
        self.alice.add(Dense(64, input_dim=(
            self.msg_len + self.key_len), activation='relu'))
        self.alice.add(Dense(64, activation='relu'))
        self.alice.add(Dense(self.msg_len, activation='sigmoid'))

        self.bob = Sequential()
        self.bob.add(Dense(64, input_dim=(
            self.msg_len + self.key_len), activation='relu'))
        self.bob.add(Dense(64, activation='relu'))
        self.bob.add(Dense(self.msg_len, activation='sigmoid'))

        self.eve = Sequential()
        self.eve.add(Dense(64, input_dim=(self.msg_len), activation='relu'))
        self.eve.add(Dense(64, activation='relu'))
        self.eve.add(Dense(self.msg_len, activation='sigmoid'))

        sgd = SGD(lr=self.learning_rate, decay=1e-6,
                  momentum=0.9, nesterov=True)
        self.alice.compile(loss='binary_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])

        self.bob.compile(loss='binary_crossentropy',
                         optimizer=sgd,
                         metrics=['accuracy'])

        self.eve.compile(loss='binary_crossentropy',
                         optimizer=sgd,
                         metrics=['accuracy'])

    # Define training
    def train(self):
        for epoch in range(self.epochs):
            print("Beginning epoch %d/%d" % (epoch+1, self.epochs))
            keys = np.random.randint(2, size=(self.batch_size, self.key_len))
            msgs = np.random.randint(2, size=(self.batch_size, self.key_len))
