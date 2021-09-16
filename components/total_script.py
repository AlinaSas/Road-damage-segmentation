"""Препроцессинг -> Аугментация -> Обучение модели дорог -> Обучение модели дефектов -> Оценка и визуализация -> Определение размеров"""
import os


from Data_Preprocessor import Preprocessor, Augmentation
from Road_models import UNET_model, DeeplabV3_mv2_model
from Evaluate_and_Inference import Evaluate_and_Inference
from Defect_area import Defect_area


project_directory = '/home/alina/PycharmProjects/roads_git/'

defect_image_directory = project_directory + 'imgs/Road_defects/PNG/FULL_SIZE/def_imgs/'
defect_mask_directory = project_directory + 'imgs/Road_defects/PNG/FULL_SIZE/def_masks/'
defect_val_image_directory = project_directory + 'imgs/Road_defects/PNG/FULL_SIZE/val_defect_imgs/'
defect_val_mask_directory = project_directory + 'imgs/Road_defects/PNG/FULL_SIZE/val_defect_mask/'

save_defect_image_directory = project_directory + 'imgs/Road_defects/PNG/imgs/'
save_defect_mask_directory = project_directory + 'imgs/Road_defects/PNG/masks/'
npy_defect_image_directory = project_directory + 'imgs/Road_defects/NPY/np_imgs/'
npy_defect_mask_directory = project_directory + 'imgs/Road_defects/NPY/np_masks/'

augmented_defect_image_directory = project_directory + 'data/Road_defects/PNG/augmentated_imgs/'
augmented_defect_mask_directory = project_directory + 'data/Road_defects/PNG/augmentated_masks/'

road_image_directory = project_directory + 'imgs/Road_surface/PNG/FULL_SIZE/def_imgs/'
road_mask_directory = project_directory + 'imgs/Road_surface/PNG/FULL_SIZE/def_masks/'
road_val_image_directory = project_directory + 'imgs/Road_defects/PNG/FULL_SIZE/val_road_imgs/'
road_val_mask_directory = project_directory + 'imgs/Road_defects/PNG/FULL_SIZE/val_road_mask/'

save_road_image_directory = project_directory + 'imgs/Road_surface/PNG/imgs/'
save_road_mask_directory = project_directory + 'imgs/Road_surface/PNG/masks/'
npy_road_image_directory = project_directory + 'imgs/Road_surface/NPY/np_imgs/'
npy_road_mask_directory = project_directory + 'imgs/Road_surface/NPY/np_masks/'

augmented_road_image_directory = project_directory + 'data/Road_surface/PNG/augmentated_imgs/'
augmented_road_mask_directory = project_directory + 'data/Road_surface/PNG/augmentated_masks/'


road_imgs = [road_image_directory + im
             for im in sorted(os.listdir(road_image_directory))]

road_masks = [road_mask_directory + m
              for m in sorted(os.listdir(road_mask_directory))]

road_val_imgs = [road_val_image_directory + im
                 for im in sorted(os.listdir(road_val_image_directory))]

road_val_masks = [road_val_mask_directory + m
                  for m in sorted(os.listdir(road_val_mask_directory))]

defect_imgs = [defect_image_directory + im
               for im in sorted(os.listdir(defect_image_directory))]

defect_masks = [defect_mask_directory + m
                for m in sorted(os.listdir(defect_mask_directory))]

defect_val_imgs = [defect_val_image_directory + im
                   for im in sorted(os.listdir(defect_val_image_directory))]

defect_val_masks = [defect_val_mask_directory + m
                    for m in sorted(os.listdir(defect_val_mask_directory))]


road_preprocess = Preprocessor(data=road_imgs, mask_data=road_masks, type='image_with_mask',
                               npy_directory=npy_road_image_directory, npy_mask_directory=npy_road_mask_directory,
                               save_directory=save_road_image_directory,
                               mask_save_directory=save_road_mask_directory,
                               crop_coordinate=[0, 1100, 3840, 2160], resize_params=[352, 1216],
                               plot=False)

val_road_preprocess = Preprocessor(data=road_val_imgs, mask_data=road_val_masks, type='image_with_mask',
                               npy_directory=npy_road_image_directory, npy_mask_directory=npy_road_mask_directory,
                               save_directory=save_road_image_directory,
                               mask_save_directory=save_road_mask_directory,
                               crop_coordinate=[0, 1100, 3840, 2160], resize_params=[352, 1216],
                               plot=False)

defects_preprocess = Preprocessor(data=defect_imgs, mask_data=defect_masks, type='image_with_mask',
                          npy_directory=npy_defect_image_directory, npy_mask_directory=npy_defect_mask_directory,
                          save_directory=save_defect_image_directory,
                          mask_save_directory=save_defect_mask_directory,
                          crop_coordinate=[0, 1100, 3840, 2160], resize_params=[352, 1216],
                          plot=False)