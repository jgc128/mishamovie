# %%
import os

import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mishamovie.utils.io import load_json
# %%
data_path = '/home/azureuser/cloudfiles/code/data/'
image_filename = os.path.join(data_path, 'frames/frame_00007.png')
faces_info_filename = os.path.join(data_path, 'faces_info/faces.json')
# %%
image = dlib.load_rgb_image(str(image_filename))
face_info = load_json(faces_info_filename)[os.path.basename(image_filename)]
# %%
plt.imshow(image)
# %%
# Extract face
# %%
buffer = 0.5
# %%
face_x1 = face_info['rect']['left']
face_x2 = face_info['rect']['left'] + face_info['rect']['width']

face_y1 = face_info['rect']['top']
face_y2 = face_info['rect']['top'] + face_info['rect']['height']
# %%
buffer_x = int((face_x2 - face_x1) * buffer)
buffer_y = int((face_y2 - face_y1) * buffer)
# %%
face_box_x1 = face_x1 - buffer_x
face_box_x2 = face_x2 + buffer_x
face_box_y1 = face_y1 - buffer_y
face_box_y2 = face_y2 + buffer_y
# %%
face = image[face_box_y1:face_box_y2, face_box_x1:face_box_x2, :]
# %%
plt.imshow(face)
# %%
# calc face coordinates on the face patch
# %%
fig, ax = plt.subplots(figsize=(10, 5))

ax.imshow(image)
ax.add_patch(
    mpatches.Rectangle(
        (face_box_x1, face_box_y1), face_box_x2 - face_box_x1, face_box_y2 - face_box_y1,
        edgecolor='red', fill=False,
    )
)

fig.tight_layout()
# %%
fig, ax = plt.subplots(figsize=(10, 5))

ax.imshow(face)
ax.add_patch(
    mpatches.Rectangle(
        (buffer_x, buffer_y), face_x2 - face_x1, face_y2 - face_y1,
        edgecolor='red', fill=False,
    )
)

for point in face_info['points']:
    point_x = point['x'] - face_info['rect']['left'] + buffer_x
    point_y = point['y'] - face_info['rect']['top'] + buffer_y
    ax.add_patch(
        mpatches.Circle(
            (point_x, point_y),
            edgecolor='green', fill=False,
            radius=1
        )
    )

fig.tight_layout()
# %%
