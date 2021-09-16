import numpy as np
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    """Класс для передачи изображеий нейросети небольшими батчами."""

    def __init__(self, list_IDs, batch_size=1, dim=(352, 1216), n_channels=3,
                 n_classes=2, shuffle=False):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.int16)
        Y = np.empty((self.batch_size, *self.dim, 1))

        for i, ID in enumerate(list_IDs_temp):
            image_path = ID
            mask_path = ID.replace('imgs', 'masks')
            img = np.load(image_path)
            mask = np.load(mask_path)

            X[i] = img
            Y[i] = mask
        return X, Y



