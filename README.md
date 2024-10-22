# ReN-UNet
Rethinking the Nested U-Net Approach: Enhancing Biomarker Segmentation with Attention Mechanisms and Multiscale Feature Fusion

Training and Testing
edit "config.txt" file to set training parameters and define folder paths

run "ren-unet.py" file in conda environmnet. it contains model, training and testing code.


Configrations

#define paths for folders that contain patches of images and masks for training

train_images_patch_dir=/mnt/hdd_2A/datasets/monuseg_patches_augm/images/
train_masks_patch_dir=/mnt/hdd_2A/datasets/monuseg_patches_augm/masks/


#define paths for numpy monuseg_test_patches_arrays that contain patches of images and masks for testing
test_images_patch_dir=/mnt/hdd_2A/datasets/monuseg_test_patches_arrays/monuseg_org_X_test.npy
test_masks_patch_dir=/mnt/hdd_2A/datasets/monuseg_test_patches_arrays/monuseg_org_y_test.npy


#define paths for folders that contain full size images and masks for testing
image_full_test_directory=/mnt/hdd_2A/datasets/monuseg_org/test/image/
mask_full_test_directory=/mnt/hdd_2A/datasets/monuseg_org/test/mask/

################################################################



#Training parameters

training=False

gpu_device=0
num_epochs=2
batch_size=6
imgz_size=256

################################################################



#Evaluation parameters

##parameters to process patches of images and masks

patch_img_size=256
patch_step_size=128

### make it false when full image size have different width and height
resize_img=True
resize_height_width=1024


##parameters to process full size images and masks

###if it is false then full size images are not scale down and evaluation takes more time
resize_full_images=True
