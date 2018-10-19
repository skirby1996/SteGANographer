class Config(object):
    def __init__(self):
        # Input and output configuration.
        self.MSG_SIZE = 16
        self.KEY_SIZE = 16

        # Training parameters.
        self.NUM_EPOCHS = 8192
        self.BATCH_SIZE = 128
        self.LEARNING_RATE = 0.0008
        self.ITERS_PER_ACTOR = 1
        self.EVE_MULTIPLIER = 2  # Train Eve 2x for every step of Alice/Bob

        # Logging parameters
        self.LOG_CHECKPOINT = 100  # Log error rate every n epochs

    def print_summary(self):
        print("Training for %d epochs with batch size of %d\n" % (self.NUM_EPOCHS, self.BATCH_SIZE))
        print("Eve trains %d times for each step of Alice and Bob\n" % (self.EVE_MULTIPLIER))
        print("msg_size = %d\nkey_size = %d\n" % (self.MSG_SIZE, self.KEY_SIZE))
        print("learning_rate = %f" % (self.LEARNING_RATE)) 
