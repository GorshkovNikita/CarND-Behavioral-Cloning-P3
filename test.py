from scipy import ndimage
import numpy as np
import scipy.misc
import imageio

center_image = ndimage.imread('../data/IMG/right_2019_04_22_07_03_12_520.jpg')
imageio.imwrite('./examples/source.jpg', center_image)
center_image_flipped = np.fliplr(center_image)
print(center_image_flipped)

imageio.imwrite('./examples/flipped.jpg', center_image_flipped)
