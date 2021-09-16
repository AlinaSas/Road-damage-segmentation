from keras.models import Model
from keras.layers import Input, BatchNormalization, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import callbacks as cb
import tensorflow as tf
from keras import optimizers
import matplotlib.pyplot as plt
from tensorflow import reduce_sum, reduce_mean, math


from numpy import random
import os
import sys


from Callbacks_lib import LossAccHistory
from Data_Generator import DataGenerator


project_directory = '/home/alina/PycharmProjects/roads_test/'

'''img_path = project_directory + 'imgs/Road_defects/NPY/np_imgs/'
mask_path = project_directory + 'imgs/Road_defects/NPY/np_masks/'
img_val_path = project_directory + 'imgs/Road_defects/NPY/val_np_imgs/'
mask_val_path = project_directory + 'imgs/Road_defects/NPY/val_np_masks/'

model_dir = project_directory + 'model_checkpoints/defects_'''


class Defect_model:
    """Класс для создания и обучения модели UNET для сегментации повреждений дорожного покрытия.
       Входные параметры:
       image_path_list - список путей к изображениям из обучающего набора;
       mask_path_list - список путей к маскам из обучающего набора;
       image_val_path_list - список путей к изображениям из валидационного набора;
       mask_val_path_list - список путей к маскам из валидационного набора;
       model_dir - директоия для сохранения модели;,
       loss_function - функция потерь (доступные варианты: binary_crossentropy, jaccard_loss, jaccard_loss2."""

    def __init__(self, image_path_list=None, mask_path_list=None,
                 image_val_path_list=None, mask_val_path_list=None, model_dir=None,
                 loss_function='jaccard_loss2', learning_rate=0.0008):
        self.image_path_list = image_path_list
        self.mask_path_list = mask_path_list
        self.image_val_path_list = image_val_path_list
        self.mask_val_path_list = mask_val_path_list
        self.model_dir = model_dir
        self.loss_function = loss_function
        self.learning_rate = learning_rate

    @staticmethod
    def Create_Model():
        """Создаётся модель UNET для сегментации повреждний дорожного покрытия"""

        input_img = Input((352, 1216, 3), name='img')

        c1 = Conv2D(32, (3, 3), padding='same')(input_img)
        c1 = BatchNormalization()(c1)
        c1 = Activation(activation='relu')(c1)
        c1 = Conv2D(32, (3, 3), padding='same')(c1)
        c1 = BatchNormalization()(c1)
        c1 = Activation(activation='relu')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(200, (3, 3), padding='same')(p1)
        c2 = BatchNormalization()(c2)
        c2 = Activation(activation='relu')(c2)
        c2 = Conv2D(200, (3, 3), padding='same')(c2)
        c2 = BatchNormalization()(c2)
        c2 = Activation(activation='relu')(c2)

        c6 = Conv2D(300, (3, 3), padding='same')(p2)
        c6 = BatchNormalization()(c6)
        c6 = Activation(activation='relu')(c6)
        c6 = Dropout(0.3)(c6)
        c6 = Conv2D(300, (3, 3), padding='same')(c6)
        c6 = BatchNormalization()(c6)
        c6 = Activation(activation='relu')(c6)
        c6 = Dropout(0.3)(c6)

        u9 = Conv2DTranspose(200, (2, 2), strides=(2, 2), padding='same')(c6)
        u9 = concatenate([u9, c2])
        c9 = Conv2D(200, (3, 3), padding='same')(u9)
        c9 = BatchNormalization()(c9)
        c9 = Activation(activation='relu')(c9)
        c9 = Conv2D(200, (3, 3), padding='same')(c9)
        c9 = BatchNormalization()(c9)
        c9 = Activation(activation='relu')(c9)

        u10 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
        u10 = concatenate([u10, c1])
        c10 = Conv2D(32, (3, 3), padding='same')(u10)
        c10 = BatchNormalization()(c10)
        c10 = Activation(activation='relu')(c10)
        c10 = Conv2D(32, (3, 3), padding='same')(c10)
        c10 = BatchNormalization()(c10)
        c10 = Activation(activation='relu')(c10)

        l = Dense(128, activation='relu')(c10)
        l = Dropout(0.3)(l)
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(l)

        model = Model(inputs=[input_img], outputs=[outputs])
        return model

    @staticmethod
    def Jaccard_loss(y_true, y_pred, smooth=10):
        intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
        sum_ = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        jd = (1 - jac) * smooth
        return tf.reduce_mean(jd)

    @staticmethod
    def Jaccard_loss2(y_true, y_pred, smooth=1):
        intersection = reduce_sum(y_true * y_pred, axis=(1, 2))
        sum_ = reduce_sum(math.square(y_true) + math.square(y_pred), axis=(1, 2))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        jd = (1 - jac) * smooth
        return reduce_mean(jd)

    def Compile_Model(self, model):
        """Компиляция модели с заданной функцией потерь и скоростью обучения.
        Во время обучения рассчитываются следующие метрики: индекс Дайса, IoU, доля верных ответов, Точность, полнота"""

        if self.loss_function == 'binary_crossentropy':
            loss = 'binary_crossentropy'
        elif self.loss_function == 'jaccard_loss':
            loss = self.Jaccard_loss
        elif self.loss_function == 'jaccard_loss2':
            loss = self.Jaccard_loss2
        else:
            print('Данная функция потерь не поддерживается.\n '
                  'Доступные функции потерь: binary_crossentropy, jaccard_loss, jaccard_loss2')
            sys.exit()

        dice = dice_coef
        IOU = iou
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss=loss,
                      metrics=[dice, IOU, tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])

    def train_model(self, epochs, batch_size):
        """Обучение модели.
        Данные подаются модели датагенератором. Каждую эпоху модель сохраняется в указанную дирректорию,
        в названии указывается номер эпохи и значение функции потерь.
        Также метод возвращает историю изменения метрик."""

        training_generator = DataGenerator(self.image_path_list, batch_size=batch_size)
        validation_generator = DataGenerator(self.image_val_path_list, batch_size=batch_size)

        model = self.Create_Model()
        self.Compile_Model(model)

        history = LossAccHistory()
        callback = [cb.ModelCheckpoint(filepath=self.model_dir + 'model_unet_df.{epoch:02d}_{loss:02f}.h5',
                                       monitor='val_loss', save_weights_only=False),
                    history]

        model.fit_generator(generator=training_generator, epochs=epochs,
                            validation_data=validation_generator, callbacks=callback)
        model.save('m_' + str(epochs) + '.h5')
        return history


def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred)
    return tf.reduce_mean((2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth))


def iou(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
    res = (intersection + smooth) / (sum_ - intersection + smooth)
    return tf.reduce_mean(res)


'''def shuffle(seed=400):
    """Перемешивает выборку с заданным параметров random.seed"""
    random.seed(seed)
    random.shuffle(imgs)
    random.seed(seed)
    random.shuffle(masks)
    random.seed(seed)
    random.shuffle(imgs_val)
    random.seed(seed)
    random.shuffle(masks_val)


imgs = [img_path + im for im in sorted(os.listdir(img_path))]
masks = [mask_path + m for m in sorted(os.listdir(mask_path))]
imgs_val = [img_val_path + im for im in sorted(os.listdir(img_val_path))]
masks_val = [mask_val_path + m for m in sorted(os.listdir(mask_val_path))]

shuffle(seed=442)

print(len(imgs))
print(imgs)
print(len(masks))
print(masks)
print(imgs_val)
print(masks_val)

rm = Defect_model(imgs, masks, imgs_val, masks_val, model_dir)
h = rm.train_model(100)'''