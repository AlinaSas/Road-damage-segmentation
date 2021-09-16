import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sn

from keras import models
from tensorflow import reduce_sum, reduce_mean, math
from tensorflow.python.keras.backend import relu

from sklearn.metrics import confusion_matrix

import keract
import cv2

import plotly.express as px
import pandas as pd

project_directory = '/home/alina/PycharmProjects/road_test/'

road_image_path = project_directory + 'data/Road_surface/PNG/imgs/'
road_mask_path = project_directory + 'data/Road_surface/PNG/masks/'

# road_model1_path = project_directory + 'models/deeplab_model_jrms.93.h5'
road_model1_path = project_directory + 'actual_model/unet_model__new_bce_07_79+23.h5'
road_model2_path = project_directory + 'actual_model/deeplab_model_jac2_1205_61+36_0.015060926787555218.h5'
# road_model2_path = project_directory + 'unet_model_unet2.68.h5'

defect_image_path = project_directory + 'data/Road_defects/PNG/road_imgs/'
defect_mask_path = project_directory + 'data/Road_defects/PNG/masks/'
defect_model_path = project_directory + 'actual_model/unet_defect_model_0.493354.h5'

save_predict_npy_path = project_directory + 'data/Predictions/NPY/'
save_predict_image_path = project_directory + 'data/Predictions/IMG/'

save_activation_directory = project_directory + 'act/'

history_plots_directory = project_directory + 'history_plot/'
loss_history = project_directory + 'metrics_checkpoints/continue_loss.npy'
vloss_history = project_directory + 'metrics_checkpoints/continue_vloss.npy'
pa_history = project_directory + 'metrics_checkpoints/continue_pa.npy'
vpa_history = project_directory + 'metrics_checkpoints/continue_vpa.npy'
dice_history = project_directory + 'metrics_checkpoints/continue_dice.npy'
vdice_history = project_directory + 'metrics_checkpoints/continue_vdice.npy'
iou_history = project_directory + 'metrics_checkpoints/continue_iou .npy'
viou_history = project_directory + 'metrics_checkpoints/continue_viou.npy'
prec_history = ''
vprec_history = ''
rec_history = ''
vrec_history = ''


class Evaluate_and_Inference:
    """Класс визуализации работы модели. Позволяет получить предсказания,
    получить метрики качества для любой из подзадач - сегменации дорожного покрытия или повреждений на нём,
    при чём может быть использованно любое число моделей с любой степенью важности каждой. Метрики рассчитыватся
    как для отдельного изображения, так и средние значения для всего набора,
    строятся графики изменения метрик на оучающей и валидационной выборке.
    Также можно создать видео, отображающее предсказания модели.
    Для отдельного изображения можно получить карту активации.
    Входные данные:
    model_type - тип модели, в зависимости от задачи сегментирвоания('road' - дорожное покртыия, 'defect' - повреждния),
    metric - булева переменная, указывающая, нужно ли вычислять метрики качества (IoU, Weighted Pixel Accuracy,
             Precision, Recall, Dice index, Pixel Accuracy, Confusion matrix).
             Метрики считаются для каждого изображения в отдельности и выводятся на экран.
             Обязательно должен быть указан путь к маскам изображений,
    coefficients - список коэффициентов значимости модели. На эти числа умножаются предсказания моделей сответственно.
                   (Если модель одна - коэффициент 1),
    save_image - булева переменная, указывающая, нужно ли сохранять результат работы моделей в виде изображения .png,
    save_npy - булева переменная, указывающая, нужно ли сохранять результат работы моделей в формате .npy,
    prediction_plot - булева переменная, указывающая, нужно ли построить график предсказаний
                      (исходное изображение, сегментированное моделью изображение),
    global_metric - булева переменная, указывающая, нужно ли вычислять средние метрики качества на всём наборе
                    (IoU, Weighted Pixel Accuracy, Precision, Recall, Dice index, Pixel Accuracy, Confusion matrix),
    video - булева переменная, указывающая, нужно ли сгенерировать видео из сегментировнных моделью изображенй,
    predicts_for_video_directory - путь с картами сегментации для генерации видео, если None, будет выполнена
                                   сегментация, при этом значение переменной save должно быть True,
    video_path - путь для сохранения видео,
    get_activation - булева переменная, указывающая, нужно ли получить и сохранить карту активации по слоям модели,
    image_for_activations - изображения, для которого будет построена карта активации,
    heatmap - булева переменная, указывающая, нужно ли строить heatmap,
    history_plot - булева переменная, указывающая, нужно ли построить графики изменения метрик в зависмости от эпохи.
                   При этом должны быть указаны пути к np.array, хранящими данные значения. Графики сохраняются
                   в формате .html
    """

    def __init__(self, model_type='road', metric=False, coefficients=[0.5, 0.5],
                 save_image=True, save_npy=True, prediction_plot=False, global_metric=True,
                 video=False, predicts_for_video_directory=None, video_path='',
                 get_activation=False, image_for_activations='', heatmap=True,
                 history_plot=False):
        global list_model_path
        self.save_image = save_image
        self.save_npy = save_npy
        self.prediction_plot = prediction_plot

        if model_type == 'road':
            list_model_path = [road_model1_path, road_model2_path]
            self.make_segmentation_and_evaluate(metric, global_metric, coefficients, model_type)
        if model_type == 'defect':
            list_model_path = [defect_model_path]
            self.make_segmentation_and_evaluate(metric, global_metric, coefficients, model_type)
        if history_plot:
            self.history()
        if video:
            if model_type == 'road':
                list_model_path = [road_model1_path, road_model2_path]
            if model_type == 'defect':
                list_model_path = [defect_model_path]
            self.render_video(video_path=video_path, coefficients=coefficients, mod=model_type,
                              predicts_for_video_directory=predicts_for_video_directory)
        if get_activation:
            self.get_activations(image_for_activations, heatmap=heatmap)

    def make_segmentation_and_evaluate(self, metric, global_metric, coefficients, mod):
        list_models = []
        for model in list_model_path:
            list_models.append(models.load_model(model,
                                                 custom_objects={'relu6': self.relu6,
                                                                 'Jaccard_loss': self.Jaccard_loss,
                                                                 'Jaccard_loss2': self.Jaccard_loss2}))
        imgs = image_paths()
        if metric:
            masks = mask_paths()
            PA, WPA, PREC, REC, DICE, IoU = {}, {}, {}, {}, {}, {}
            yt, yp = [], []
            for im, m in zip(imgs, masks):
                print('Начинаем предсказание для ', im, '...')
                predictions, test_image = self.create_predictions(im, list_models, coefficients)
                predictions = predictions.flatten()
                predictions = np.array([1 if i >= 0.55 else 0 for i in predictions])
                prediction_mask = predictions.reshape(image_size[0], image_size[1], 1)

                self.save(save_predict_npy_path + im.split('/')[-1] + '.npy',
                          prediction_mask, test_image,
                          save_predict_image_path + '_' + mod + '_' + str(len(list_models)) + '_'
                          + im.split('/')[-1].split('.')[0], mod=mod)

                mask = self.load_image_as_np(m, type='mask', mod=mod)

                yp.append(list(predictions))
                yt.append(list(mask.flatten()))

                print('Значение метрик качества для ', im, ':')
                WPA[im], PREC[im], REC[im], DICE[im], PA[im], IoU[im] = self.calculate_metrics(mask, prediction_mask,
                                                                                               conf=True)
                print('\nIoU = ', IoU[im], '\nWPA = ', WPA[im], '\nPrecision = ', PREC[im], '\nRecall = ',
                      REC[im], '\nDice index = ', DICE[im], '\nPA = ', PA[im])

            if global_metric:
                print('Значение метрик качества для всего набора:')
                cm = confusion_matrix(np.reshape(yt, (len(yt) * len(yt[0]))),
                                      np.reshape(yp, (len(yp) * len(yp[0]))), normalize='true')
                sn.set(font_scale=1.4)
                sn.heatmap(cm, annot=True, cmap="cool")
                plt.show()
                metrics = self.global_metrics(PA, IoU, WPA, PREC, REC, DICE)
                print('PA_mean = ', metrics[0], 'IoU_mean = ', metrics[1], 'WPA_mean = ', metrics[2],
                      'Precision_mean = ', metrics[3], 'Recall_mean = ', metrics[4], 'Dice_mean = ', metrics[5])
        else:
            for im in imgs:
                print('Начинаем предсказание для ', im, '...')
                predictions, test_image = self.create_predictions(im, list_models, coefficients)
                predictions = predictions.flatten()
                predictions = np.array([1 if i >= 0.5 else 0 for i in predictions])
                prediction_mask = predictions.reshape(image_size[0], image_size[1])

                self.save(save_predict_npy_path + im.split('/')[-1] + '.npy',
                          prediction_mask, test_image,
                          save_predict_image_path + '_' + mod + '_' + str(len(list_models)) + '_'
                          + im.split('/')[-1].split('.')[0], mod=mod)

    def save(self, npy_path, prediction_mask, test_image, image_path, mod='road'):
        if self.save_npy:
            np.save(save_predict_npy_path + npy_path.split('/')[-1] + '.npy',
                    prediction_mask)
        if self.save_image:
            if mod == 'road':
                new_image = self.create_new_road_image(test_image, prediction_mask)
            else:
                new_image = self.create_new_defect_image(test_image, prediction_mask)
            self.save_new_image(new_image, image_path)
        if self.prediction_plot:
            self.pred_plots(prediction_mask, test_image)

    def create_predictions(self, im, list_models, coefficients):
        test_image = np.asarray(Image.open(im))
        if test_image.shape != (image_size[0], image_size[1], 3):
            print('Размер изображения не подходит для модели и будет изменён')
            test_image = np.asarray(Image.open(im).resize((image_size[1], image_size[0]), 0).convert('RGB'))

        predictions = 0
        for i, model in enumerate(list_models):
            model_predictions = self.get_predict(test_image, model)
            predictions += coefficients[i] * model_predictions
        return predictions, test_image

    def load_image_as_np(self, im, type='im', mod='road'):
        if im.find('.npy') != -1:
            image = np.load(im)
        else:
            image = np.asarray(Image.open(im).convert('RGB'))
            if type == 'mask':
                if mod == 'defect':
                    image = self.mask_binarization(image, [255, 0, 100]).reshape(352, 1216, 1)
                else:
                    image = self.mask_binarization(image, [0, 0, 255]).reshape(352, 1216, 1)
        return image

    @staticmethod
    def Jaccard_loss(y_true, y_pred, smooth=1):
        intersection = reduce_sum(y_true * y_pred, axis=(1, 2))
        sum_ = reduce_sum(y_true + y_pred, axis=(1, 2))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        jd = (1 - jac) * smooth
        return reduce_mean(jd)

    @staticmethod
    def Jaccard_loss2(y_true, y_pred, smooth=1):
        intersection = reduce_sum(y_true * y_pred, axis=(1, 2))
        sum_ = reduce_sum(math.square(y_true) + math.square(y_pred), axis=(1, 2))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        jd = (1 - jac) * smooth
        return reduce_mean(jd)

    @staticmethod
    def calculate_metrics(pt, pp, conf):
        TP, FP, TN, FN = 0, 0, 0, 0
        for t, p in zip(pt, pp):
            for elt, elp in zip(t, p):
                if elt[0] == 1 and elp[0] == 1:
                    TP += 1
                elif elt[0] == 1 and elp[0] == 0:
                    FN += 1
                elif elt[0] == 0 and elp[0] == 1:
                    FP += 1
                elif elt[0] == 0 and elp[0] == 0:
                    TN += 1

        PA = (TP + TN) / (TP + TN + FN + FP)
        WPA = ((1 - (TP + FN) / (TP + TN + FP + FN)) * TP + ((TP + FN) / (TP + TN + FP + FN)) * TN + 0.0001) / \
              ((1 - (TP + FN) / (TP + TN + FP + FN)) * TP + ((TP + FN) / (TP + TN + FP + FN)) * TN
               + (1 - (TP + FN) / (TP + TN + FP + FN)) * FP + ((TP + FN) / (TP + TN + FP + FN)) * FN + 0.0001)
        PREC = (TP + 0.0001) / (TP + FP + 0.0001)
        REC = (TP + 0.0001) / (TP + FN + 0.0001)
        DICE_index = (2 * TP + 0.0001) / (2 * TP + FP + FN + 0.0001)
        IoU = TP / (TP + FP + FN + 0.0001)

        if conf:
            cm = confusion_matrix(pt.flatten(), pp.flatten(), normalize='true')
            sn.set(font_scale=1.4)
            sn.heatmap(cm, annot=True, cmap="cool")
            plt.show()
        return [WPA, PREC, REC, DICE_index, PA, IoU]

    @staticmethod
    def global_metrics(PIXEL_ACCURACY, IoU, WPA, PREC, REC, DICE):
        PA_mean = np.mean(np.array(list(PIXEL_ACCURACY.values())))
        IoU_mean = np.mean(np.array(list(IoU.values())))
        WPA_mean = np.mean(np.array(list(WPA.values())))
        Prec_mean = np.mean(np.array(list(PREC.values())))
        Rec_mean = np.mean(np.array(list(REC.values())))
        Dice_mean = np.mean(np.array(list(DICE.values())))
        return [PA_mean, IoU_mean, WPA_mean, Prec_mean, Rec_mean, Dice_mean]

    @staticmethod
    def relu6(x):
        return relu(x, max_value=6)

    @staticmethod
    def get_predict(image, model):
        my_preds = model.predict(np.expand_dims(image, 0))
        return my_preds.reshape(image_size[0], image_size[1])

    @staticmethod
    def create_new_road_image(image, prediction_image):
        new_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.int8)
        for i in range(len(prediction_image)):
            for j in range(len(prediction_image[i])):
                if prediction_image[i][j] == 1:
                    new_image[i][j] = image[i][j]
                else:
                    new_image[i][j] = [0, 0, 0]
        return new_image

    @staticmethod
    def create_new_defect_image(image, prediction_image):
        new_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.int8)
        for i in range(len(prediction_image)):
            for j in range(len(prediction_image[i])):
                if prediction_image[i][j] == 1:
                    new_image[i][j] = [0, 0, 255]
                else:
                    new_image[i][j] = image[i][j]
        return new_image

    @staticmethod
    def save_new_image(new_image, new_image_path):
        new_img = Image.fromarray(new_image, 'RGB')
        new_img.save(new_image_path + '.png', 'PNG')

    @staticmethod
    def mask_binarization(mask, color):
        M = mask.copy()
        MR = np.zeros((image_size[0], image_size[1], 1), dtype=np.uint8)

        M[M == color] = 1
        M[M != 1] = 0

        for i in range(image_size[0]):
            for j in range(image_size[1]):
                if M[i][j][0] != 0:

                    MR[i][j] = 1
                else:
                    MR[i][j] = 0

        return MR.reshape((image_size[0], image_size[1]))

    @staticmethod
    def pred_plots(prediction_mask, real_mask):
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(prediction_mask)
        ax[0].set_title('Prediction')
        ax[1].imshow(real_mask)
        ax[1].set_title('Ground truth')
        plt.show()

    def history(self):
        loss, val_loss = np.load(loss_history), np.load(vloss_history)
        pa, val_pa = np.load(pa_history), np.load(vpa_history)
        dice, val_dice = np.load(dice_history), np.load(vdice_history)
        iou, val_iou = np.load(iou_history), np.load(viou_history)
        self.history_plot(loss, val_loss, ['loss', 'val_loss'], 'LOSS')
        self.history_plot(pa, val_pa, ['pa', 'val_pa'], 'Pixel accuracy')
        self.history_plot(dice, val_dice, ['dice', 'val_dice'], 'Dice Index')
        self.history_plot(iou, val_iou, ['iou', 'val_iou'], 'IoU')

    @staticmethod
    def history_plot(history, val_history, columns, title):
        data_loss = pd.DataFrame(data=np.transpose(np.array([history, val_history])), columns=columns)
        fig = px.line(data_loss, x=data_loss.index, y=columns, template='plotly_dark')
        fig.update_layout(title=title + '<br>best train ' + columns[0] + ' : ' + str(round(min(history), 3))
                          + '  best ' + columns[1] + ' : ' + str(round(min(val_history), 3)),
                          xaxis_title="epochs", yaxis_title=columns[0],
                          font=dict(family="Roboto",
                                    size=14,
                                    color="white"),
                          margin=dict(l=0, r=0, t=120, b=0))

        fig.show()
        fig.write_html(history_plots_directory + title + '.html')

    def get_activations(self, image, save_directory=save_activation_directory, heatmap=True):
        print('get activation')
        model = models.load_model(road_model1_path,
                                  custom_objects={'relu6': self.relu6, 'Jaccard_loss': self.Jaccard_loss})
        activations = keract.get_activations(model, image, auto_compile=True)
        print('display')
        if heatmap:
            keract.display_heatmaps(activations, image, save=True, directory=save_directory)
        else:
            keract.display_activations(activations, save=True, directory=save_activation_directory)

    def render_video(self, video_path, coefficients, mod, image_directory='',
                     predicts_for_video_directory=None):
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (image_size[1], image_size[0]))
        if predicts_for_video_directory is None:
            self.make_segmentation_and_evaluate(metric=False, global_metric=False, coefficients=coefficients,
                                                mod=mod)
            predicts_for_video_directory = save_predict_image_path
        imgs = [predicts_for_video_directory + im for im in sorted(os.listdir(image_directory))]
        print('Рендеринг видео...')
        for im in imgs:
            i = cv2.imread(im)
            video.write(i)
        video.release()


def image_paths():
    return [defect_image_path + im for im in sorted(list(os.listdir(defect_image_path)))][35:]


def mask_paths():
    return [defect_mask_path + m for m in sorted(list(os.listdir(defect_mask_path)))][35:]


image_size = (352, 1216)
i = Evaluate_and_Inference(model_type='defect', metric=True, coefficients=[1.0],
                           video=False, predicts_for_video_directory=None,
                           video_path='')
