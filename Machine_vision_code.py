import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('/path/to/img', cv2.IMREAD_GRAYSCALE)
T = 120 
_, bin_img = cv2.threshold(img, T, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((10, 10), np.uint8)
closing = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(closing, kernel, iterations=6)

num_labels, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=4, ltype=cv2.CV_32S)
# returns N labels, with statistics and centroids in an array [0, N-1] where 0 represents the background label
print(f'Found {num_labels-1} objects')

plt.figure(figsize=(10, 10))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

for (p1x, p1y, size_x, size_y, _) in stats[1:]:
    print(f'Object inside the rectangle with coordinates ({p1x},{p1y}), ({p1x+size_x}, {p1y+size_y})')
    cv2.rectangle(img_rgb, (p1x,p1y), (p1x+size_x, p1y+size_y), (0,0,255), 2)

for (cx, cy) in centroids[1:]:
    cx, cy = int(cx), int(cy)
    print(f'Centroid: ({cx},{cy})')
    cv2.circle(img_rgb, (cx,cy), 2, (191,40,0), 2)
    cv2.putText(img_rgb, f"({cx},{cy})" , (cx-40, cy-15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 125), 1, cv2.LINE_AA)

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Annotated Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(eroded, cmap='gray')
plt.title(f'Global Threshold (T={T})')
plt.axis('off')
plt.tight_layout()
plt.show()
