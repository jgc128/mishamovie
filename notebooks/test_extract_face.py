#%%
import os

import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mishamovie.utils.io import load_json
#%%
data_path = '/home/azureuser/cloudfiles/code/data/'
image_filename = os.path.join(data_path, 'frames/frame_00007.png')
faces_info_filename = os.path.join(data_path, 'faces_info/faces.json')
#%%
image = dlib.load_rgb_image(str(image_filename))
face_info = load_json(faces_info_filename)[os.path.basename(image_filename)]
#%%
plt.imshow(image)
# %%
# Extract face
#%%
buffer = 0.5
#%%
face_x1 = face_info['rect']['top']
face_x2 = face_info['rect']['top'] + face_info['rect']['height']

face_y1 = face_info['rect']['left']
face_y2 = face_info['rect']['left'] + face_info['rect']['width']
#%%
buffer_x = int((face_x2 - face_x1) * buffer)
buffer_y = int((face_y2 - face_y1) * buffer)
#%%
face = image[face_x1 - buffer_x:face_x2 + buffer_x, face_y1 - buffer_y:face_y2 + buffer_y, :]
# %%
plt.imshow(face)
# %%
# calc face coordinates on the face patch
#%%
rect_x = buffer_x
rect_y = buffer_y
#%%
fig, ax = plt.subplots(figsize=(10, 5))

ax.imshow(face)
ax.add_patch(
    mpatches.Rectangle(
        (rect_x, rect_y), face_x2 - face_x1, face_y2- face_y1,
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
