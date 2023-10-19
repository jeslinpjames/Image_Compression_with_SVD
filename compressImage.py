import cv2
import numpy as np
from svd import SVD
import matplotlib.pyplot as plt
import os

image_path = "D:/git/Image_Compression_with_SVD/img_2.jpeg"

image = cv2.imread(image_path)

image = image.astype(np.uint8)

R = image[:, :, 0]
G = image[:, :, 1]
B = image[:, :, 2]

svd_R = SVD(R)
svd_G = SVD(G)
svd_B = SVD(B)

rank = 200
compressed_R = svd_R.reconstruct_matrix(rank)
compressed_G = svd_G.reconstruct_matrix(rank)
compressed_B = svd_B.reconstruct_matrix(rank)

# Stack the channels and convert to 8-bit unsigned integer format
compressed_image = np.stack([compressed_R, compressed_G, compressed_B], axis=-1).astype(np.uint8)

# Save the compressed image to a file
compressed_image_path = "D:/git/Image_Compression_with_SVD/compressed_image.jpg"
cv2.imwrite(compressed_image_path, compressed_image)

original_image_size = os.path.getsize(image_path) / 1024
compressed_image_size = os.path.getsize(compressed_image_path) / 1024

# Display the images and their sizes
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title(f"Original Image ({original_image_size:.2f} KB)")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Compressed Image (Rank {rank}, {compressed_image_size:.2f} KB)")
plt.imshow(compressed_image)
plt.axis('off')

plt.show()
