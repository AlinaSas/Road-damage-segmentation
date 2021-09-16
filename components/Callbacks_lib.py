import keras
import numpy as np


project_directory = '/home/alina/PycharmProjects/road_test/'


class LossAccHistory(keras.callbacks.Callback):
    """Класс для вычисления и сохранения значения метрик(IoU,Precision, Recall, Dice index, Pixel Accuracy)
    в конце каждой эпохи на обучающей и валидационной выборках"""

    def __init__(self):
        super().__init__()
        self.losses = []
        self.val_loss = []
        self.iou = []
        self.dice = []
        self.pa = []
        self.prec = []
        self.rec = []
        self.val_iou = []
        self.val_dice = []
        self.val_pa = []
        self.val_prec = []
        self.val_rec = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        self.iou.append(logs.get('iou'))
        self.val_iou.append(logs.get('val_iou'))

        self.dice.append(logs.get('dice_coef'))
        self.val_dice.append(logs.get('val_dice_coef'))

        self.pa.append(logs.get('binary_accuracy'))
        self.val_pa.append(logs.get('val_binary_accuracy'))

        self.prec.append(logs.get('precision_9'))
        self.val_prec.append(logs.get('val_precision_9'))

        self.rec.append(logs.get('recall_9'))
        self.val_rec.append(logs.get('val_recall_9'))

        np.save(project_directory + 'metrics_checkpoints/continue_pa.npy', self.pa)
        np.save(project_directory + 'metrics_checkpoints/continue_vpa.npy', self.val_pa)
        np.save(project_directory + 'metrics_checkpoints/continue_iou.npy', self.iou)

        np.save(project_directory + 'metrics_checkpoints/continue_viou.npy', self.val_iou)
        np.save(project_directory + 'metrics_checkpoints/continue_dice.npy', self.dice)
        np.save(project_directory + 'metrics_checkpoints/continue_vdice.npy', self.val_dice)
        np.save(project_directory + 'metrics_checkpoints/continue_loss.npy', self.losses)
        np.save(project_directory + 'metrics_checkpoints/continue_vloss.npy', self.val_loss)




