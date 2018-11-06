import os
import random
import numpy as np
from PIL import Image


class Dataset(object):
    def __init__(self, cfg, type):
        path = os.path.abspath("")
        path = os.path.join(path, "datasets")
        path = os.path.join(path, cfg.DATASET_NAME)
        self.path = os.path.join(path, type)
        self.img_list = os.listdir(self.path)
        self.img_ix = random.randint(0, len(self.img_list) - 1)

    def get_batch(self, batch_size, img_size):
        '''
        Arguments:
            batch_size: Number of images to collect
            img_size: Pixel height/width of image
        Returns:
            batch: A [bach_size, img_size, img_size, 3]
                array of (0., 1.) floating falues
        '''

        batch = np.zeros(shape=(batch_size, img_size, img_size, 3))

        for ix in range(batch_size):
            img_path = os.path.join(self.path, self.img_list[self.img_ix])
            with Image.open(img_path) as img:
                w, h = img.size
                x = random.randint(0, w - img_size)
                y = random.randint(0, h - img_size)
                img = img.crop(box=(x, y, x + img_size, y + img_size))
                # img.show()
                data = np.asarray(img, dtype='float32')
                # print("in dataset\n", data)
                data /= 255.
                if data.shape != (img_size, img_size, 3):
                    print(img_path)
                    print(data.shape)
                batch[ix] = data

            self.img_ix += 1
            if self.img_ix == len(self.img_list):
                self.img_ix = 0

        return batch
