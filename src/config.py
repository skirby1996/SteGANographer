class Config(object):
    def __init__(self):
        # Input and output configuration.
        self.IMG_SIZE = 32
        self.NUM_CHANNELS = 3

        self.MSG_SIZE = 32
        self.KEY_SIZE = 32

        # Training parameters.
        self.NUM_EPOCHS = 25000
        self.BATCH_SIZE = 64

        self.LEARNING_RATE = 0.0008
        self.ITERS_PER_ACTOR = 1
        self.ALICE_MULTIPLIER = 3  # Train alice_bob_optimizer x times each epoch
        self.BOB_MULTIPLIER = 1  # Train bob_optimizer x times each epoch

        # Dataset parameters
        self.DATASET_NAME = "office"

        # Logging parameters
        self.LOG_CHECKPOINT = 25  # Log error rate every n epochs

        # File parameters
        self.MODEL_NAME = "StegoNet_32_v2"

    def print_summary(self):
        print("Model %s training for %d epochs with batch size of %d\n" %
              (self.MODEL_NAME, self.NUM_EPOCHS, self.BATCH_SIZE))
        print("Bob trains %d times for each step of Alice and Bob\n" %
              (self.BOB_MULTIPLIER))
        print("img_shape = (%d, %d, %d)\n" %
              (self.IMG_SIZE, self.IMG_SIZE, self.NUM_CHANNELS))
        print("msg_size = %d\nkey_size = %d\n" %
              (self.MSG_SIZE, self.KEY_SIZE))
        print("learning_rate = %f" % (self.LEARNING_RATE))
