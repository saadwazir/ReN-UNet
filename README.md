
# ReN-UNet

**Rethinking the Nested U-Net Approach: Enhancing Biomarker Segmentation with Attention Mechanisms and Multiscale Feature Fusion**

## Training and Testing

1. Edit the `config.txt` file to set training parameters and define folder paths.
2. Run the `ren-unet.py` file in a conda environment. It contains the model, training, and testing code.

---

## Configurations

### Paths for training
- Define paths for folders that contain patches of images and masks for training.

```ini
train_images_patch_dir=/mnt/hdd_2A/datasets/monuseg_patches_augm/images/
train_masks_patch_dir=/mnt/hdd_2A/datasets/monuseg_patches_augm/masks/
```

### Paths for testing
- Define paths for numpy arrays that contain patches of images and masks for testing.

```ini
test_images_patch_dir=/mnt/hdd_2A/datasets/monuseg_test_patches_arrays/monuseg_org_X_test.npy
test_masks_patch_dir=/mnt/hdd_2A/datasets/monuseg_test_patches_arrays/monuseg_org_y_test.npy
```

- Define paths for folders that contain full-size images and masks for testing.

```ini
image_full_test_directory=/mnt/hdd_2A/datasets/monuseg_org/test/image/
mask_full_test_directory=/mnt/hdd_2A/datasets/monuseg_org/test/mask/
```

---

## Training Parameters

```ini
training=False
gpu_device=0
num_epochs=2
batch_size=6
imgz_size=256
```

---

## Evaluation Parameters

### Parameters for processing patches of images and masks:

```ini
patch_img_size=256
patch_step_size=128
```

- Set `resize_img=False` if full image sizes have different width and height.

```ini
resize_img=True
resize_height_width=1024
```

### Parameters for processing full-size images and masks:

- If `resize_full_images=False`, full-size images are not scaled down, but evaluation takes more time.

```ini
resize_full_images=True
```

---

### Notes:
- Make sure the paths and parameters are configured correctly in `config.txt` before running the code.
