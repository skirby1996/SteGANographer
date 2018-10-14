class Config(object):
    def __init__(self):
        # Input and output configuration.
        self.MSG_SIZE = 16
        self.KEY_SIZE = 16

        # Training parameters.
        self.NUM_EPOCHS = 1024
        self.BATCH_SIZE = 128
        self.LEARNING_RATE = 0.0008
        self.ITERS_PER_ACTOR = 1
        self.EVE_MULTIPLIER = 2  # Train Eve 2x for every step of Alice/Bob

        # Logging parameters
        self.LOG_CHECKPOINT = 64  # Log error rate every n epochs
