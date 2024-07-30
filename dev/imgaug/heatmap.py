import imageio
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.heatmaps import HeatmapsOnImage

ia.seed(1)

# Load an example image (uint8, 128x128x3).
image = ia.quokka(size=(128, 128), extract="square")

# Create an example depth map (float32, 128x128).
# Here, we use a simple gradient that has low values (around 0.0)
# towards the left of the image and high values (around 50.0)
# towards the right. This is obviously a very unrealistic depth
# map, but makes the example easier.
depth = np.linspace(0, 50, 128).astype(np.float32)  # 128 values from 0.0 to 50.0
depth = np.tile(depth.reshape(1, 128), (128, 1))    # change to a horizontal gradient

# We add a cross to the center of the depth map, so that we can more
# easily see the effects of augmentations.
depth[64-2:64+2, 16:128-16] = 0.75 * 50.0  # line from left to right
depth[16:128-16, 64-2:64+2] = 1.0 * 50.0   # line from top to bottom

# Convert our numpy array depth map to a heatmap object.
# We have to add the shape of the underlying image, as that is necessary
# for some augmentations.
depth = HeatmapsOnImage(
    depth, shape=image.shape, min_value=0.0, max_value=50.0)

# To save some computation time, we want our models to perform downscaling
# and hence need the ground truth depth maps to be at a resolution of
# 64x64 instead of the 128x128 of the input image.
# Here, we use simple average pooling to perform the downscaling.
depth = depth.avg_pool(2)

# Define our augmentation pipeline.
seq = iaa.Sequential([
    iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
    iaa.Sharpen((0.0, 1.0)),       # sharpen the image
    iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects heatmaps)
    iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects heatmaps)
], random_order=True)

# Augment images and heatmaps.
images_aug = []
heatmaps_aug = []
for _ in range(5):
    images_aug_i, heatmaps_aug_i = seq(image=image, heatmaps=depth)
    images_aug.append(images_aug_i)
    heatmaps_aug.append(heatmaps_aug_i)

# We want to generate an image of original input images and heatmaps
# before/after augmentation.
# It is supposed to have five columns:
# (1) original image,
# (2) augmented image,
# (3) augmented heatmap on top of augmented image,
# (4) augmented heatmap on its own in jet color map,
# (5) augmented heatmap on its own in intensity colormap.
# We now generate the cells of these columns.
#
# Note that we add a [0] after each heatmap draw command. That's because
# the heatmaps object can contain many sub-heatmaps and hence we draw
# command returns a list of drawn sub-heatmaps.
# We only used one sub-heatmap, so our lists always have one entry.
cells = []
for image_aug, heatmap_aug in zip(images_aug, heatmaps_aug):
    cells.append(image)                                                     # column 1
    cells.append(image_aug)                                                 # column 2
    cells.append(heatmap_aug.draw_on_image(image_aug)[0])                   # column 3
    cells.append(heatmap_aug.draw(size=image_aug.shape[:2])[0])             # column 4
    cells.append(heatmap_aug.draw(size=image_aug.shape[:2], cmap=None)[0])  # column 5

# Convert cells to grid image and save.
grid_image = ia.draw_grid(cells, cols=5)
imageio.imwrite("example_heatmaps.jpg", grid_image)