class Config(object):
    def __init__(self):
        # Input and output configuration.
        self.IMG_SIZE = 32
        self.NUM_CHANNELS = 3

        self.MSG_SIZE = 32
        self.KEY_SIZE = 32

        # Training parameters.
        self.NUM_EPOCHS = 10000
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.0008
        self.ITERS_PER_ACTOR = 1
        self.EVE_MULTIPLIER = 2  # Train Eve 2x for every step of Alice/Bob

        # Dataset parameters
        self.DATASET_NAME = "office"

        # Logging parameters
        self.LOG_CHECKPOINT = 25  # Log error rate every n epochs

        # File parameters
        self.MODEL_NAME = "Dev"

    def print_summary(self):
        print("Model %s training for %d epochs with batch size of %d\n" %
              (self.MODEL_NAME, self.NUM_EPOCHS, self.BATCH_SIZE))
        print("Eve trains %d times for each step of Alice and Bob\n" %
              (self.EVE_MULTIPLIER))
        print("img_shape = (%d, %d, %d)\n" %
              (self.IMG_SIZE, self.IMG_SIZE, self.NUM_CHANNELS))
        print("msg_size = %d\nkey_size = %d\n" %
              (self.MSG_SIZE, self.KEY_SIZE))
        print("learning_rate = %f" % (self.LEARNING_RATE))
