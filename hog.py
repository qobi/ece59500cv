import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import exposure

# Load and convert the image to grayscale
image = imread('house00.jpg')
grayscaled_image = rgb2gray(image)

# Compute HOG features and get the visualization
fd, hog_image = hog(grayscaled_image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, channel_axis=None)

# Use exposure.rescale_intensity to improve the contrast of the HOG image
# was in_range=(0, 10)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.1))

# Display the original and HOG images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(image)
ax1.set_title('Original Image')
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('HOG Visualization')
plt.tight_layout()
plt.show()
