import os
import utils
import skimage
import numpy as np

DUMMY = 'NULL'
class BatchGenerator:
    """ Loads images in a directory into batches 

        Args:
            batch_size: Number of items per batch
            image_h: Height of image
            image_w: Width of image
            image_dir: Image directory to load batches from
            max_batches: Max number of batches to load into memory at any given moment
            valid: Whether or not this is a validation set
            logging: Logging object
            batch_index: Batch index to start loading from
    """ 
    def __init__(self, batch_size, image_h, image_w, image_dir=DUMMY, max_batches=100, valid=False, logging=None, batch_index=0):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.max_batches = max_batches 
        if logging:
            logging.info("setting up batches %d at a time starting from %d" % (self.max_batches, batch_index))
        else:
            print("setting up batches %d at a time starting from %d" % (self.max_batches, batch_index))
        self.image_h = image_h
        self.image_w = image_w
        self.logging = logging
        file_list = os.listdir(image_dir) if image_dir != DUMMY else []
        self.last_load = batch_index if not valid else len(file_list)-1       
        self.index = 0
        self.batches = None
        self.valid = valid
        self.load_batches()

    def load_batches(self):
        """ Load max_batches batches into memory """
        is_new = False
        if self.batches:
            batches = self.batches
        else:
            is_new = True
            batches = []
            self.batches = batches
        image_dir = self.image_dir
        batch_size = self.batch_size
        image_h = self.image_h
        image_w = self.image_w
        file_list = os.listdir(image_dir) if image_dir != DUMMY else []
        n = self.last_load 
        if self.logging:
            self.logging.info("file index %d" % n)
        else:
            print("file index %d" % n)
        for b in xrange(self.max_batches):
            if is_new:
                arr = np.zeros((batch_size, image_h, image_w, 3))
                batches.append(arr)
            else:
                arr = batches[b]
            if image_dir != DUMMY:
                i = 0
                while i < batch_size:
                    file_name = file_list[n]
                    try:
                        image = utils.load_image(os.path.join(image_dir,file_name), image_h, image_w)
                        arr[i] = image
                        i += 1
                    except:
                        pass
                    n += 1 if not self.valid else -1
        self.last_load = n

    def get_batch(self):
        """ Returns the next batch. Starts loading the next set of batches into memory 
        if we reach the end 

        """
        batch = self.batches[self.index]
        self.index += 1
        if self.index >= len(self.batches):            
            self.index = 0
            self.load_batches()
        return np.array(batch)

    def get_last_load(self):
        return self.last_load
