class Config(object):
    def __init__(self):
        # Input and output configuration.
        self.IMG_SIZE = 32  # Width/Height of image region to encode
        self.NUM_CHANNELS = 3  # Number of channels in image region to encode

        # Training parameters.
        self.NUM_EPOCHS = 9000  # Total number of epochs to train
        self.BATCH_SIZE = 64  # Number of batches in a training batch
        self.TEST_BATCH_SIZE = 64  # Number of batches in a eval batch

        self.LEARNING_RATE = 0.0008  # Learning rate for training, should be low
        self.ITERS_PER_ACTOR = 1  # Number of training rounds per epoch
        self.ALICE_MULTIPLIER = 1  # Train alice_bob_optimizer x times each epoch
        self.BOB_MULTIPLIER = 1  # Train bob_optimizer x times each epoch
        self.EVE_MULTIPLIER = 1  # Train eve_optimizer x times each epoch

        # Dataset parameters
        self.DATASET_NAME = "office"  # Folder name of image dataset

        # Logging parameters
        self.LOG_CHECKPOINT = 25  # Log error rate every n epochs

        # File parameters
        self.MODEL_NAME = "Dev_32_v5"  # Folder name of model

    def print_summary(self):
        print("Model %s training for %d epochs with batch size of %d\n" %
              (self.MODEL_NAME, self.NUM_EPOCHS, self.BATCH_SIZE))
        print("img_shape = (%d, %d, %d)\n" %
              (self.IMG_SIZE, self.IMG_SIZE, self.NUM_CHANNELS))
        print("learning_rate = %f" % (self.LEARNING_RATE))
