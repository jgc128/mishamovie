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
faceboxes_info_filename = os.path.join(data_path, 'faces_info/faceboxes_info.json')
# %%
image = dlib.load_rgb_image(str(image_filename))
face_aged = dlib.load_rgb_image(str(face_aged_filenane))
facebox_info = load_json(faceboxes_info_filename)[filename]
# %%
plt.imshow(image)
# %%
plt.imshow(face_aged)
# %%
fig, ax = plt.subplots(figsize=(10, 5))

ax.imshow(image)
ax.add_patch(
    mpatches.Rectangle(
        (facebox_info['x1'], facebox_info['y1']),
        facebox_info['x2'] - facebox_info['x1'], facebox_info['y2'] - facebox_info['y1'],
        edgecolor='red', fill=False,
    )
)

fig.tight_layout()
# %%
# image[facebox_info['y1']:facebox_info['y2'], facebox_info['x1']:facebox_info['x2'], :] = face_aged
# %%
plt.imshow(image)
# %%
face_mask = np.full_like(face_aged, 255)
face_center_coord = (
    facebox_info['x1'] + (facebox_info['x2'] - facebox_info['x1']) // 2,
    facebox_info['y1'] + (facebox_info['y2'] - facebox_info['y1']) // 2,
)

image_with_face = cv2.seamlessClone(face_aged, image, face_mask, face_center_coord, cv2.NORMAL_CLONE)
# %%
plt.imshow(image_with_face)
# %%