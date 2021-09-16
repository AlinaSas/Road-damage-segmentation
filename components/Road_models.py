from numpy import random
import os
import sys

from keras.models import Model
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import callbacks as cb
from keras import optimizers
from keras.layers import Input, Conv2D, BatchNormalization, Activation, DepthwiseConv2D, \
    Add, GlobalAveragePooling2D, Lambda, Concatenate, Dropout, Dense, UpSampling2D, Conv2DTranspose

import tensorflow as tf
from tensorflow.python.keras import backend as K

from Callbacks_lib import LossAccHistory
from Data_Generator import DataGenerator

from Path import img_path, mask_path, img_val_path, mask_val_path, model_dir

class UNET_model:
    """Класс для создания и обучения модели UNET для сегментации дорожного покрытия.
    Входные параметры:
    image_path_list - список путей к изображениям из обучающего набора;
    mask_path_list - список путей к маскам из обучающего набора;
    image_val_path_list - список путей к изображениям из валидационного набора;
    mask_val_path_list - список путей к маскам из валидационного набора;
    model_dir - директоия для сохранения модели;,
    loss_function - функция потерь (доступные варианты: binary_crossentropy, jaccard_loss, jaccard_loss2."""

    def __init__(self, image_path_list, mask_path_list, image_val_path_list, mask_val_path_list, model_dir,
                 loss_function='binary_crossentropy', learning_rate=0.0006):

        self.image_path_list = image_path_list
        self.mask_path_list = mask_path_list
        self.image_val_path_list = image_val_path_list
        self.mask_val_path_list = mask_val_path_list
        self.model_dir = model_dir
        self.loss_function = loss_function
        self.learning_rate = learning_rate

    @staticmethod
    def Create_Model():
        """Создаётся модель UNET для сегментации дорожного покрытия"""

        input_img = Input((352, 1216, 3), name='img')

        c1 = Conv2D(64, (3, 3), padding='same')(input_img)
        c1 = BatchNormalization()(c1)
        c1 = Activation(activation='relu')(c1)
        c1 = Conv2D(64, (3, 3), padding='same')(c1)
        c1 = BatchNormalization()(c1)
        c1 = Activation(activation='relu')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(128, (3, 3), padding='same')(p1)
        c2 = BatchNormalization()(c2)
        c2 = Activation(activation='relu')(c2)
        c2 = Conv2D(128, (3, 3), padding='same')(c2)
        c2 = BatchNormalization()(c2)
        c2 = Activation(activation='relu')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(256, (3, 3), padding='same')(p2)
        c3 = BatchNormalization()(c3)
        c3 = Activation(activation='relu')(c3)
        c3 = Conv2D(256, (3, 3), padding='same')(c3)
        c3 = BatchNormalization()(c3)
        c3 = Activation(activation='relu')(c3)
        c3 = Dropout(0.25)(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(512, (3, 3), padding='same')(p3)
        c4 = BatchNormalization()(c4)
        c4 = Activation(activation='relu')(c4)
        c4 = Conv2D(512, (3, 3), padding='same')(c4)
        c4 = BatchNormalization()(c4)
        c4 = Activation(activation='relu')(c4)
        c4 = Dropout(0.25)(c4)
        p4 = MaxPooling2D((2, 2))(c4)

        c6 = Conv2D(1024, (3, 3), padding='same')(p4)
        c6 = BatchNormalization()(c6)
        c6 = Activation(activation='relu')(c6)
        c6 = Dropout(0.3)(c6)
        c6 = Conv2D(1024, (3, 3), padding='same')(c6)
        c6 = BatchNormalization()(c6)
        c6 = Activation(activation='relu')(c6)
        c6 = Dropout(0.3)(c6)

        u7 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c4])
        c7 = Conv2D(512, (3, 3), padding='same')(u7)
        c7 = BatchNormalization()(c7)
        c7 = Activation(activation='relu')(c7)
        c7 = Conv2D(512, (3, 3), padding='same')(c7)
        c7 = BatchNormalization()(c7)
        c7 = Activation(activation='relu')(c7)
        c7 = Dropout(0.25)(c7)

        u8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c3])
        c8 = Conv2D(256, (3, 3), padding='same')(u8)
        c8 = BatchNormalization()(c8)
        c8 = Activation(activation='relu')(c8)
        c8 = Conv2D(256, (3, 3), padding='same')(c8)
        c8 = BatchNormalization()(c8)
        c8 = Activation(activation='relu')(c8)
        c8 = Dropout(0.25)(c8)

        u9 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c2])
        c9 = Conv2D(128, (3, 3), padding='same')(u9)
        c9 = BatchNormalization()(c9)
        c9 = Activation(activation='relu')(c9)
        c9 = Conv2D(128, (3, 3), padding='same')(c9)
        c9 = BatchNormalization()(c9)
        c9 = Activation(activation='relu')(c9)

        u10 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c9)
        u10 = concatenate([u10, c1])
        c10 = Conv2D(64, (3, 3), padding='same')(u10)
        c10 = BatchNormalization()(c10)
        c10 = Activation(activation='relu')(c10)
        c10 = Conv2D(64, (3, 3), padding='same')(c10)
        c10 = BatchNormalization()(c10)
        c10 = Activation(activation='relu')(c10)

        l = Dense(64, activation='relu')(c10)
        l = Dropout(0.3)(l)
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(l)

        model = Model(inputs=[input_img], outputs=[outputs])
        return model

    def Compile_Model(self, model):
        """Компиляция модели с заданной функцией потерь и скоростью обучения.
        Во время обучения рассчитываются следующие метрики: индекс Дайса, IoU, доля верных ответов, Точность, полнота"""

        if self.loss_function == 'binary_crossentropy':
            loss = 'binary_crossentropy'
        elif self.loss_function == 'jaccard_loss':
            loss = Jaccard_loss
        elif self.loss_function == 'jaccard_loss_2':
            loss = Jaccard_loss2
        else:
            print('Данная функция потерь не поддерживается.\n '
                  'Доступные функции потерь: binary_crossentropy, jaccard_loss, jaccard_loss2')
            sys.exit()

        dice = dice_coef
        IOU = iou
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss=loss,
                      metrics=[dice, IOU, tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])

    def train_model(self, epochs):
        """Обучение модели.
        Данные подаются модели датагенератором. Каждую эпоху модель сохраняется в указанную дирректорию,
        в названии указывается номер эпохи и значение функции потерь.
        Также метод возвращает историю изменения метрик."""

        training_generator = DataGenerator(self.image_path_list)
        validation_generator = DataGenerator(self.image_val_path_list)

        model = self.Create_Model()
        self.Compile_Model(model)

        history = LossAccHistory()
        callback = [cb.ModelCheckpoint(filepath=self.model_dir + '_' + self.loss_function + '{epoch:02d}{loss}.h5',
                                       monitor='loss', save_weights_only=False),
                    history]

        model.fit(x=training_generator, epochs=epochs,
                  validation_data=validation_generator,
                  callbacks=callback)
        model.save('m_' + str(epochs) + '.h5')
        return history


class DeeplabV3_mv2_model:
    """Класс для создания и обучения модели Deeplab v3 с энкодером mobilenet v2 для сегментации дорожного покрытия.
       Входные параметры:
       image_path_list - список путей к изображениям из обучающего набора;
       mask_path_list - список путей к маскам из обучающего набора;
       image_val_path_list - список путей к изображениям из валидационного набора;
       mask_val_path_list - список путей к маскам из валидационного набора;
       model_dir - директоия для сохранения модели;
       loss_function - функция потерь (доступные варианты: binary_crossentropy, jaccard_loss, jaccard_loss2;
       alpha - коэффицент раздутия в mv2 блоках"""
    def __init__(self,  image_path_list, mask_path_list, image_val_path_list, mask_val_path_list, model_dir,
                 loss_function='binary_crossentropy', alpha=1, learning_rate=0.0006):
        self.alpha = alpha
        self.image_path_list = image_path_list
        self.mask_path_list = mask_path_list
        self.image_val_path_list = image_val_path_list
        self.mask_val_path_list = mask_val_path_list
        self.model_dir = model_dir
        self.loss_function = loss_function
        self.learning_rate = learning_rate

    def Create_Model(self):
        """Создаётся архитектура Deeplab v3 с вызовом метода построения mobilenet v2"""

        input = Input((352, 1216, 3))
        x = self.MobileNetV2(input)

        b4 = GlobalAveragePooling2D()(x)
        b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
        b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
        b4 = Conv2D(256, (1, 1), padding='same', use_bias=False)(b4)
        b4 = BatchNormalization()(b4)
        b4 = Activation(activation='relu')(b4)

        size_before = tf.keras.backend.int_shape(x)
        b4 = UpSampling2D(size=size_before[1:3])(b4)

        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False)(x)
        b0 = BatchNormalization()(b0)
        b0 = Activation(activation='relu')(b0)

        x = Concatenate()([b4, b0])

        x = Conv2D(256, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        x = Dropout(0.15)(x)

        x = Conv2D(2, (3, 3), padding='same')(x)
        x = Activation(activation='relu')(x)
        size_before3 = tf.keras.backend.int_shape(input)
        x = UpSampling2D(size=(int(size_before3[1] / size_before[1]), int(size_before3[2] / size_before[2])))(x)

        x = Conv2D(1, (1, 1), padding='same')(x)
        x = Activation(activation='sigmoid')(x)

        model = Model(input, x)
        return model

    def MobileNetV2(self, input):
        """Создаётся общая архитектура mobilenet v2, с вызовом метода создания блоков"""

        x = Conv2D(24, kernel_size=3, strides=(2, 2), padding='same', use_bias=False)(input)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.relu6)(x)

        x = self.mnv2_block(inputs=x, filters=16, stride=1, expansion=1, skip_connection=False)

        x = self.mnv2_block(inputs=x, filters=32, stride=2, expansion=6, skip_connection=False)
        x = self.mnv2_block(inputs=x, filters=32, stride=1, expansion=6, skip_connection=True)

        x = self.mnv2_block(inputs=x, filters=32, stride=2, expansion=6, skip_connection=False)
        x = self.mnv2_block(inputs=x, filters=32, stride=1, expansion=6, skip_connection=True)
        x = self.mnv2_block(inputs=x, filters=32, stride=1, expansion=6, skip_connection=True)
        x = Dropout(0.25)(x)

        x = self.mnv2_block(inputs=x, filters=64, stride=1, expansion=6, skip_connection=False)
        x = self.mnv2_block(inputs=x, filters=64, stride=1, rate=2, expansion=6, skip_connection=True)
        x = self.mnv2_block(inputs=x, filters=64, stride=1, rate=2, expansion=6, skip_connection=True)
        x = self.mnv2_block(inputs=x, filters=64, stride=1, rate=2, expansion=6, skip_connection=True)
        x = Dropout(0.25)(x)

        x = self.mnv2_block(inputs=x, filters=96, stride=1, rate=2, expansion=6, skip_connection=False)
        x = self.mnv2_block(inputs=x, filters=96, stride=1, rate=2, expansion=6, skip_connection=True)
        x = self.mnv2_block(inputs=x, filters=96, stride=1, rate=2, expansion=6, skip_connection=True)
        x = Dropout(0.25)(x)

        x = self.mnv2_block(inputs=x, filters=160, stride=1, rate=2, expansion=6, skip_connection=False)
        x = self.mnv2_block(inputs=x, filters=160, stride=1, rate=4, expansion=6, skip_connection=True)
        x = Dropout(0.25)(x)
        x = self.mnv2_block(inputs=x, filters=160, stride=1, rate=4, expansion=6, skip_connection=True)
        x = Dropout(0.25)(x)

        x = self.mnv2_block(inputs=x, filters=256, stride=1, rate=4, expansion=6, skip_connection=False)
        x = Dropout(0.25)(x)
        x = self.mnv2_block(inputs=x, filters=256, stride=1, rate=6, expansion=6, skip_connection=True)
        x = Dropout(0.25)(x)
        x = self.mnv2_block(inputs=x, filters=256, stride=1, rate=6, expansion=6, skip_connection=True)
        x = Dropout(0.25)(x)

        x = self.mnv2_block(inputs=x, filters=512, stride=1, rate=6, expansion=6, skip_connection=False)
        x = Dropout(0.25)(x)

        return x

    def mnv2_block(self, inputs, expansion, stride, filters, skip_connection, rate=1):
        """Создаётся блок mobilenet v2"""

        in_channels = inputs.shape[-1]
        pointwise_conv_filters = int(filters * self.alpha)
        x = inputs
        if expansion != 1:
            x = Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation(tf.nn.relu6)(x)

        x = DepthwiseConv2D(kernel_size=3, strides=stride, use_bias=False, padding='same', dilation_rate=(rate, rate))(
            x)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.relu6)(x)

        x = Conv2D(pointwise_conv_filters, kernel_size=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)

        if skip_connection:
            return Add()([inputs, x])

        return x

    def Compile_Model(self, model):
        """Компиляция модели с заданной функцией потерь и скоростью обучения.
        Во время обучения рассчитываются следующие метрики: индекс Дайса, IoU, доля верных ответов, Точность, полнота"""

        if self.loss_function == 'binary_crossentropy':
            loss = 'binary_crossentropy'
        elif self.loss_function == 'jaccard_loss':
            loss = Jaccard_loss
        elif self.loss_function == 'jaccard_loss_2':
            loss = Jaccard_loss2
        else:
            print('Данная функция потерь не поддерживается.\n '
                  'Доступные функции потерь: binary_crossentropy, jaccard_loss, jaccard_loss2')
            sys.exit()

        dice = dice_coef
        IOU = iou
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss=loss,
                      metrics=[dice, IOU, tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])

    def train_model(self, batch_size, epochs):
        """Обучение модели.
        Данные подаются модели датагенератором. Каждую эпоху модель сохраняется в указанную дирректорию,
        в названии указывается номер эпохи и значение функции потерь.
        Также метод возвращает историю изменения метрик."""

        training_generator = DataGenerator(self.image_path_list, batch_size=batch_size, dim=dim)
        validation_generator = DataGenerator(self.image_val_path_list, batch_size=batch_size, dim=dim)

        model = self.Create_Model()
        self.Compile_Model(model)

        history = LossAccHistory()
        callback = [cb.ModelCheckpoint(filepath=self.model_dir + '_' + self.loss_function + '{epoch:02d}{loss}.h5',
                                       monitor='loss', save_weights_only=False),
                    history]

        model.fit(x=training_generator, epochs=epochs,
                  validation_data=validation_generator,
                  callbacks=callback)
        model.save('m_' + str(epochs) + '.h5')
        return history

    @staticmethod
    def relu6(x):
        return K.relu(x, max_value=6)


def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred)
    return tf.reduce_mean((2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth))


def iou(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
    res = (intersection + smooth) / (sum_ - intersection + smooth)
    return tf.reduce_mean(res)


def Jaccard_loss(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd = (1 - jac) * smooth
    return tf.reduce_mean(jd)


def Jaccard_loss2(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    sum_ = tf.reduce_sum(tf.math.square(y_true) + tf.math.square(y_pred), axis=(1, 2))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd = (1 - jac) * smooth
    return tf.reduce_mean(jd)


def shuffle(seed=400):
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
m = 'unet'
dim = (352, 1216)

if len(imgs) == len(masks) and len(imgs_val) == len(masks_val):
    if m == 'unet':
        print('Начало обучения модели UNET')
        rm = UNET_model(imgs, masks, imgs_val, masks_val, model_dir)
        h = rm.train_model(batch_size=6, epochs=100)
    elif m == 'deeplab':
        print('Начало обучения модели DeepLab')
        rm = DeeplabV3_mv2_model(imgs, masks, imgs_val, masks_val, model_dir)
        h = rm.train_model(batch_size=3, epochs=100)
else:
    print('В тренировочном или валидационном наборе данных разное количество файлов. Обучение невозможно.')

