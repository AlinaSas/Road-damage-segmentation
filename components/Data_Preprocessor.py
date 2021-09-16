import os
import numpy as np
from subprocess import call
import matplotlib.pyplot as plt

import albumentations as A
from PIL import Image

project_directory = '/home/alina/PycharmProjects/roads_git/'

video_path = project_directory + '/video/video/video_0_00:00:16.mp4'
save_video_cut_directory = project_directory + 'video/cut/'
save_storyboards_directory = project_directory + 'video/storyboards/'
image_directory = project_directory + 'data/Road_defects/PNG/FULL_SIZE/def_imgs/'
mask_directory = project_directory + 'data/Road_defects/PNG/FULL_SIZE/def_masks/'
save_image_directory = project_directory + 'data/Road_defects/PNG/imgs/'
save_mask_directory = project_directory + 'data/Road_defects/PNG/masks/'
npy_image_directory = ''
npy_mask_directory = ''
augmented_image_directory = project_directory + 'data/Road_defects/PNG/augmentated_imgs/'
augmented_mask_directory = project_directory + 'data/Road_defects/PNG/augmentated_masks/'


class Preprocessor:
    """Класс для предварительной обработки данных для обучения.
    Предобработка происходит для видов следующих входных данных:
    отдельные изображения, изображения с соответствующими масками, видео.
    На выходе получаются изображения с подходящими для модели размерами, которые могут быть сохранены в форматах .png, .npy.
    Входные параметры:
    data - список путей к избражениям;
    mask_data - список путей к маскам изображений (указывается только если type='image_with_mask');
    type - вид входных данных ('image', 'image_with_mask', 'video');
    npy_directory - директория для сохранения изображений в формате .npy;
    npy_mask_directory - директория для сохранения масок в формате .npy;
    save_directory - директория для сохранения изображений в формате .png;
    mask_save_directory - директория для сохранения масок в формате .png;
    crop_coordinate - список из 4х координат для обрезки: [x0, y0, x1, y1];
    resize_params - список из списков из 2х параметров, обознающих конечный размер изображения: [высота, ширина];
    object_color - цвет в формате RGB, соответствующий исследуемому объекту.
                   Для дороги: [0, 0, 255], для повреждений:  [255,0, 100];
    cut_videos - булева переменная, которая указывает, нужно ли вырезать фрагмент видео;
    cut_params - список из 2х параметров, обознающих начало интересующей части видео в формате 'hh:mm:ss'
                 и продлжительность интерсующего промежутка в секундах (формат str);
    save_cut_directory - директория для сохранения отрывков видео;
    storyboards_directory - булева переменная, которая указывает, нужно ли делать раскадровку видео;
    storyboards_prepr - булева переменная, которая указывает, нужно ли делать предобработку полученных кадров
                        (т.е. обрабатывать как 'image');
    thinning_coeff - указывает, с какой частотой оставлять кадры видео (н-р, при значении 5, будет оставлен каждый 5ый кадр).
                     Используется для удаления похожих кадров. Если равен 0, то кадры не прореживаются."""

    def __init__(self, data, mask_data='', type='image',
                 npy_directory=None, npy_mask_directory=None, save_directory=None, mask_save_directory=None, plot=False,
                 crop_coordinate=None, resize_params=None, object_color=None,
                 cut_videos=False, cut_params=None, save_cut_directory=None,
                 storyboards_directory=None, storyboards_prepr=False,
                 thinning_coeff=0):

        if object_color is None:
            object_color = [255, 0, 100]
        if cut_params is None:
            cut_params = []
        if crop_coordinate is None:
            crop_coordinate = []
        if type == 'image':
            if data == []:
                print('Дирректория с данными пустая')
            self.image_preprocess(data, crop_coordinate, resize_params, save_directory, npy_directory, plot)
        elif type == 'image_with_mask':
            print(type)
            self.image_with_mask_preprocess(data, mask_data, crop_coordinate, resize_params,
                                            save_directory, mask_save_directory, object_color,
                                            npy_directory, npy_mask_directory, plot)
        elif type == 'video':
            self.video_preprocess(data, cut_videos, cut_params, save_cut_directory,
                                  storyboards_directory, storyboards_prepr, thinning_coeff,
                                  crop_coordinate, resize_params, save_directory, npy_directory, plot)

    def video_preprocess(self, data, cut_videos, cut_params, save_cut_directory, storyboards_directory,
                         storyboards_prepr, thinning_coeff,
                         crop_coordinate, resize_params, save_directory, npy_directory):
        """При необходимости,из видео вырезаются фрагменты, разбиваются на кадры,
        которые прореживаются с заданным прмежутком. После, обрабатываются как 'image'"""

        video_path = data
        if cut_videos:
            video_path = []
            for i, params in enumerate(cut_params):
                print('Происходит обрезка видео...')
                print(params)
                video = self.cut_video(video_path, params[0], params[1], i, save_cut_directory)
                video_path.append(video)
        if storyboards_directory:
            for video in video_path:
                print('Происходит раскадровка видео...')
                video_path = self.create_storyboards(video, storyboards_directory)
        if thinning_coeff != 0:
            for i, img in enumerate(video_path):
                if i % thinning_coeff == 0:
                    next()
                else:
                    os.remove(img)
        if storyboards_prepr:
            print('Происходит пепроцессинг кадров...')
            storyboards = [video_path + '/' + v for v in sorted(os.listdir(video_path))]
            self.image_preprocess(storyboards, crop_coordinate, resize_params,
                                  save_directory, npy_directory)

    def image_with_mask_preprocess(self, data, mask_data, crop_coordinate, resize_params,
                                   save_directory, mask_save_directory, object_color,
                                   npy_directory, npy_mask_directory, plot):
        """ Препроцессинг изображения и маски позволяет обрезать и изменять размеры(сжимать,расширять)
            изображения и маски с одинаковыми параметрами, сохранить изображение и маску в формате .png,
            а также преобразовать в np.array и сохранять в .npy.
            ! Имена изображения и маски должны совпадать
            """

        for i, image_data in enumerate(data):
            print('Начием препроцессинг для', image_data, '...')
            image = self.image_open(image_data)
            mask = self.image_open(mask_data[i])

            if plot:
                real_image = image.copy()
                real_mask = mask.copy()

            dim = self.get_dim(image)
            dim_mask = self.get_dim(mask)

            if dim == dim_mask:
                if crop_coordinate:
                    crop = self.make_crop(crop_coordinate, dim, image, mask)
                    image = next(crop)
                    mask = next(crop)

                if resize_params:
                    dim_mask = resize_params
                    res = self.make_resize(resize_params, dim, image, mask)
                    image = next(res)
                    mask = next(res)
            else:
                print('Размерности изображения и маски не совпадают. Во избежание ошибок, '
                      'обработайте изображеия и маски по отдельности.')

            im_name = image_data.split('/')[-1].split('.')[0]
            m_name = mask_data[i].split('/')[-1].split('.')[0]
            if save_directory:
                print('Происходит сохранение изображения и маски....')
                self.image_save(save_directory, image, im_name)
                self.image_save(mask_save_directory, mask, m_name)
                print('Изображение и маска сохранены в формате .npy')

            if npy_directory and npy_mask_directory:
                print('Происходит сохранение изображения и маски в .npy')
                np.save(npy_directory + im_name + '.npy', image)
                self.save_mask_in_npy(mask, npy_mask_directory, m_name, object_color, dim_mask)
                print('Изображение и маска сохранены в формате .npy')

            if plot:
                self.preprocess_plots_image_with_mask(real_image, image, real_mask, mask)

    def image_preprocess(self, image_path_list, crop_coordinate, resize_params, save_directory, npy_directory, plot):
        """Препроцессинг изображения позволяет обрезать изображение, измененить размеры,
        сохранить изображение в формате .png, а также преобразовать в np.array и сохранить в .npy"""

        for image_path in image_path_list:
            image = self.image_open(image_path)
            dim = self.get_dim(image)

            if plot:
                real_image = image.copy()

            if crop_coordinate:
                crop = self.make_crop(crop_coordinate, dim, image)
                image = next(crop)

            if resize_params:
                res = self.make_resize(resize_params, dim, image)
                image = next(res)

            image_name = image_path.split('/')[-1].split('.')[0]
            if save_directory is not None:
                print('Происходит сохранение изображения....')
                self.image_save(save_directory, image, image_name)
                print('Изображение сохранено')
            if npy_directory is not None:
                print('Происходит сохранение изображения в .npy')
                np.save(npy_directory + image_name + '.npy', image)
                print('Изображение сохранено в формате .npy')
            if plot:
                print('Строится график')
                self.preprocess_plots(real_image, image)

    def make_crop(self, crop_coordinate, dim, *data):
        x, y = crop_coordinate[2] - crop_coordinate[0], crop_coordinate[3] - crop_coordinate[1]

        if dim[1] <= x and dim[0] <= y:
            print('Обрезка невозможна, так как исходные параметры изображения или маски меньше заданных')
            for elem in data:
                yield elem
        elif dim[1] > x and dim[0] <= y:
            print('Обрезка возможна только по ширине')
            for elem in data:
                yield self.crop_image(elem, crop_coordinate[0], 0, crop_coordinate[2], dim[0])
        elif dim[1] <= x and dim[0] > y:
            print('Обрезка возможна только по высоте')
            for elem in data:
                yield self.crop_image(elem, 0, crop_coordinate[1], dim[1], crop_coordinate[3])
        else:
            print('Происходит обрезка изображения')
            for elem in data:
                yield self.crop_image(elem, crop_coordinate[0], crop_coordinate[1], crop_coordinate[2],
                                       crop_coordinate[3])

    def make_resize(self, resize_params, dim, *data):
        if dim[0] <= resize_params[0] or dim[1] <= resize_params[1]:
            print('Размеры исходного изображения меньше требуемого, изображение и маска расширяются')
            for elem in data:
                yield self.resize(elem, resize_params)
        else:
            print('Происходит сжатие изображения и маски')
            for elem in data:
                yield self.resize(elem, resize_params)

    @staticmethod
    def image_save(diretory, img, img_name):
        if not os.path.exists(diretory):
            os.mkdir(diretory)
        img.save(diretory + img_name + '.png', "PNG")

    @staticmethod
    def save_mask_in_npy(M, npy_mask_directory, m_name, object_color, dim_mask):
        """Сохраняет маску в бинарном виде, где 1 - пиксель интересующего объекта (дорога или треина), 0 - фон"""

        M = np.array(M)
        MR = np.zeros((dim_mask[0], dim_mask[1], 1), dtype=np.uint8)
        M[M == object_color] = 1
        M[M != 1] = 0
        for i in range(dim_mask[0]):
            for j in range(dim_mask[1]):
                if M[i][j][2] == 1:

                    MR[i][j] = 1
                else:
                    MR[i][j] = 0

        if not os.path.exists(npy_mask_directory):
            os.mkdir(npy_mask_directory)
        np.save(npy_mask_directory + m_name + '.npy', MR)

    @staticmethod
    def create_storyboards(video, storyboards_directory):
        video_name = video.split('/')[-1].split('.mp4')[0]
        new_directory = storyboards_directory + '/' + video_name
        os.mkdir(new_directory)
        video_to_storyboards = "ffmpeg -i " + video + ' ' + new_directory + '/' + video_name + '_' + 'img%03d.png'
        call(video_to_storyboards.split(), shell=False)
        return new_directory

    @staticmethod
    def cut_video(video_in, start, seconds, i, cut_video_path):
        save_path = cut_video_path + 'video_' + str(i) + '_' + start + '.mp4'
        cut_video_command = 'ffmpeg -i ' + video_in + ' -ss ' + start + ' -t ' + seconds + ' -c:v libx264 -qp 16 ' \
                            + save_path
        call(cut_video_command.split(), shell=False)
        return save_path

    @staticmethod
    def get_dim(img):
        return [img.size[1], img.size[0]]

    @staticmethod
    def image_open(img):
        image = Image.open(img)
        return image

    @staticmethod
    def crop_image(image, x0, y0, x, y):
        return image.crop((x0, y0, x, y))

    @staticmethod
    def resize(img, resize_params):
        return img.resize((resize_params[1], resize_params[0]), 0)

    @staticmethod
    def preprocess_plots(real_image, preprocess_image):
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(real_image)
        ax[0].set_title('real_image')
        ax[1].imshow(preprocess_image)
        ax[1].set_title('preprocess_image')
        plt.show()

    @staticmethod
    def preprocess_plots_image_with_mask(real_image, preprocess_image, real_mask, preprocess_mask):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0][0].imshow(real_image)
        ax[0][0].set_title('real_image')
        ax[0][1].imshow(preprocess_image)
        ax[0][1].set_title('preprocess_image')
        ax[1][0].imshow(real_mask)
        ax[1][0].set_title('real_mask')
        ax[1][1].imshow(preprocess_mask)
        ax[1][1].set_title('preprocess_mask')
        plt.show()


class Augmentation:
    """Класс для расширения набора данных, путём зеркального отражения и цветовых преобразований имеющихся.
    Входные параметры:
    image_directory - директория с изозбражениями;
    mask_directory - директория с масками;
    aug_image_directory - директория для сохранения преобразованных изображений;
    aug_masks_directory - директория для сохранения преобразованных масок;
    plot - булева функция, указывающия нужно ли вывести исходное и преобразованное изображение и маску на графике.
    """
    def __init__(self, images, masks, aug_image_directory, aug_masks_directory, plot=False):
        for image_path, mask_path in zip(images, masks):
            image, mask = self.prepare_data(image_path, mask_path)
            image_augmented = self.augmentator(image)
            self.save_data(image_augmented, mask,
                           aug_image_directory + 'aug_' + image_path.split('/')[-1],
                           aug_masks_directory + 'aug_' + mask_path.split('/')[-1])
            if plot:
                self.pred_plots(image_path, mask_path, image_augmented, mask)

    @staticmethod
    def augmentator(im_new):
        """Метод, реализующий цветовые трансформации изображений.
        Всего используются три вида преобразований
        (Гамма-коррекция, изменение значений по цветовым каналам RGB, а также яркость, контрастность и насыщенность)
         с различными параметрами, вероятность каждого из преобразований 80%,
         что позволяет получать изображения с различными цветовыми характеристиками."""
        transformation = A.Compose([A.augmentations.transforms.RandomGamma(gamma_limit=(80, 135),
                                                                           eps=None, always_apply=False,
                                                                           p=0.8),
                                    A.augmentations.transforms.RGBShift(r_shift_limit=0, g_shift_limit=3,
                                                                        b_shift_limit=3, always_apply=False,
                                                                        p=0.5),
                                    A.augmentations.transforms.ColorJitter(brightness=[0.95, 1.5], contrast=(1.3, 1.5),
                                                                           saturation=(0.4, 1.8), always_apply=False,
                                                                           p=0.7)], p=0.8)

        transform = transformation(image=im_new)
        transformed_image = transform['image']
        return transformed_image

    @staticmethod
    def save_data(image, mask, image_path, mask_path):
        t_im = Image.fromarray(image, 'RGB')
        t_m = Image.fromarray(mask, 'RGB')
        t_im.save(image_path)
        t_m.save(mask_path)

    @staticmethod
    def pred_plots(image, mask, aug_image, aug_mask):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0][0].imshow(image)
        ax[0][0].set_title('image')
        ax[0][1].imshow(mask)
        ax[0][1].set_title('mask')
        ax[1][0].imshow(aug_image)
        ax[1][0].set_title('augmented image')
        ax[1][1].imshow(aug_mask)
        ax[1][1].set_title('augmentede mask')
        plt.show()

    @staticmethod
    def prepare_data(image, mask):
        im, m = Image.open(image), Image.open(mask)
        im_new, m_new = im.transpose(Image.FLIP_LEFT_RIGHT), m.transpose(Image.FLIP_LEFT_RIGHT)
        im_new, m_new = np.array(im_new, dtype='uint8'), np.array(m_new, dtype='uint8')
        return im_new, m_new


'''imgs = [image_directory + im
        for im in sorted(os.listdir(image_directory))]

masks = [mask_directory + m
         for m in sorted(os.listdir(mask_directory))]

prepr = Preprocessor(data=imgs, mask_data=masks, type='image',
                     crop_coordinate=[0, 1100, 3840, 2160], resize_params=[352, 1216],
                     save_directory=None,
                     mask_save_directory=None)
'''
'''Aug = Augmentation(images=imgs,
                   masks=masks,
                   aug_image_directory=augmented_image_directory,
                   aug_masks_directory=augmented_mask_directory)'''

'''prepr = Preprocessor(data=video_path, type='video',
                        crop_coordinate=[0, 1100, 3840, 2160], resize_params=[1216, 352],
                        cut_videos=False, cut_params=[['00:00:00', str(817)]],
                        save_cut_directory=save_video_cut_directory,
                        storyboards_directory=save_storyboards_directory, storyboards_prepr=False,
                        save_directory=save_image_directory)'''
