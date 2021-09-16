project_directory = '/home/alina/PycharmProjects/roads_git/'

# Data_preprocessor

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

# Road_models

img_path = project_directory + 'data/Road_surface/NPY/np_imgs/'
mask_path = project_directory + 'data/Road_surface/NPY/np_masks/'
img_val_path = project_directory + 'data/Road_surface/NPY/val_np_imgs/'
mask_val_path = project_directory + 'data/Road_surface/NPY/val_np_masks/'
model_dir = project_directory + 'model_checkpoints/road_'

# Defect_model

def_img_path = project_directory + 'imgs/Road_defects/NPY/np_imgs/'
def_mask_path = project_directory + 'imgs/Road_defects/NPY/np_masks/'
def_img_val_path = project_directory + 'imgs/Road_defects/NPY/val_np_imgs/'
def_mask_val_path = project_directory + 'imgs/Road_defects/NPY/val_np_masks/'

def_model_dir = project_directory + 'model_checkpoints/defects_'

# Evaluate_and_Inference

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

# Defect_area

image_for_defect_area_calc = project_directory + 'imgs/Road_defects/NPY/np_masks/104___video_only_img11241.npy'
image_for_proportion_calc = project_directory + 'imgs/Road_surface/NPY/val_np_masks/104___video_only_img135.npy'