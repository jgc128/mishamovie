# %%
import os

import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mishamovie.utils.io import load_json
# %%
data_path = '/home/azureuser/cloudfiles/code/data/'
filename = 'frame_00181.png'

image_filename = os.path.join(data_path, 'frames', filename)
face_aged_filenane = os.path.join(data_path, 'faces_aged', filename)
face_filenane = os.path.join(data_path, 'faces', filename)
faceboxes_info_filename = os.path.join(data_path, 'faces_info/faceboxes_info.json')
faces_info_filename = os.path.join(data_path, 'faces_info/faces.json')
# %%
image = dlib.load_rgb_image(str(image_filename))
face = dlib.load_rgb_image(str(face_filenane))
face_aged = dlib.load_rgb_image(str(face_aged_filenane))
facebox_info = load_json(faceboxes_info_filename)[filename]
face_info = load_json(faces_info_filename)[filename]
# %%
plt.imshow(image)
# %%
plt.imshow(face)
# %%
plt.imshow(face_aged)
# %%
fig, ax = plt.subplots(figsize=(10, 5))

ax.imshow(image)

# face box
ax.add_patch(
    mpatches.Rectangle(
        (facebox_info['x1'], facebox_info['y1']),
        facebox_info['x2'] - facebox_info['x1'], facebox_info['y2'] - facebox_info['y1'],
        edgecolor='blue', fill=False,
    )
)

# face detection
ax.add_patch(
    mpatches.Rectangle(
        (face_info['rect']['left'], face_info['rect']['top']),
        face_info['rect']['width'], face_info['rect']['height'],
        edgecolor='red', fill=False,
    )
)

# face points
for point in face_info['points']:
    point_x = point['x']
    point_y = point['y']
    ax.add_patch(
        mpatches.Circle(
            (point_x, point_y),
            edgecolor='green', fill=False,
            radius=1
        )
    )

fig.tight_layout()
# %%
fig, ax = plt.subplots(figsize=(10, 5))

ax.imshow(face)
ax.add_patch(
    mpatches.Rectangle(
        (face_info['rect']['left'] - facebox_info['x1'], face_info['rect']['top'] - facebox_info['y1'], ),
        face_info['rect']['width'], face_info['rect']['height'],
        edgecolor='red', fill=False,
    )
)

for point in face_info['points']:
    point_x = point['x'] - facebox_info['x1']
    point_y = point['y'] - facebox_info['y1']
    ax.add_patch(
        mpatches.Circle(
            (point_x, point_y),
            edgecolor='green', fill=False,
            radius=1
        )
    )
fig.tight_layout()
# %%
face_landmarks_array = np.array([
    [p['x'] - facebox_info['x1'], p['y'] - facebox_info['y1']]
    for p in face_info['points']
])
face_hull = cv2.convexHull(face_landmarks_array, returnPoints=True)[:, 0, :]
# %%
fig, ax = plt.subplots(figsize=(10, 5))

ax.imshow(face)

for x, y in face_hull:
    ax.add_patch(mpatches.Circle((x, y), edgecolor='green', fill=False, radius=1))

fig.tight_layout()
# %%
face_mask = np.full_like(face_aged, 0)
cv2.fillConvexPoly(face_mask, face_hull, color=(255, 255, 255))
# %%
plt.imshow(face_mask)
# %%
kernel_size = face_info['rect']['width'] // 10
face_mask_dilated = cv2.dilate(face_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
# %%
plt.imshow(face_mask_dilated)
# %%
# %%
contours, _ = cv2.findContours(face_mask_dilated[:, :, 0] * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
contours = np.concatenate(contours, axis=0)[:, 0, :]
br_x, br_y, br_w, br_h = cv2.boundingRect(contours)
center = (
    facebox_info['x1'] + br_x + br_w // 2,
    facebox_info['y1'] + br_y + br_h // 2,
)
image_with_face = cv2.seamlessClone(face_aged, image, face_mask_dilated, center, cv2.NORMAL_CLONE)
# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(image_with_face)
fig.tight_layout()
# %%
