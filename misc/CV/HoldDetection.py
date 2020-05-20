from skimage import measure
from skimage import filters
import numpy as np
import matplotlib.pyplot as plt
import imageio
import scipy.ndimage as ndimage  
from skimage.morphology import closing, square, watershed
from skimage.measure import label, regionprops


print('Starting...')

image = imageio.imread('./Images/BoardCutout.png')
print('Loaded image')
print(image.shape)

image = image[:,:, 1]
#print(g.shape)

image = np.pad(image, pad_width=10, mode='constant', constant_values=0)
image = closing(image, square(3))

##### BUILD MASK
# Find outer contours
outerContours = measure.find_contours(image, 0.2)
mask = np.zeros_like(image, dtype='bool')

for n, contour in enumerate(outerContours):
    mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
    
mask = ndimage.binary_fill_holes(mask)

# Find holds
holds = measure.find_contours(board, 0.8, mask=mask)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(segmentation, cmap='gray')

for n, contour in enumerate(holds):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
