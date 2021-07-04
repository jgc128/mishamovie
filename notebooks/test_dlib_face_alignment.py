# Based on https://github.com/davisking/dlib/blob/master/python_examples/face_alignment.py
# %%
import os

import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# %%
model_path = '/home/azureuser/cloudfiles/code/Users/jgc128/dlib-models/'
data_path = '/home/azureuser/cloudfiles/code/data/'
face_detector_path = os.path.join(model_path, 'mmod_human_face_detector.dat')
shape_predictor_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks_GTX.dat')
image_path = os.path.join(data_path, 'frames/frame_00181.png')
# %%
detector = dlib.cnn_face_detection_model_v1(face_detector_path)
shape_predictor = dlib.shape_predictor(shape_predictor_path)
# %%
image = dlib.load_rgb_image(image_path)  # image.shape (540, 1024, 3)
# %%
plt.imshow(image)
# %%
detected_faces = detector(image, 1)
# %%
plt.imshow(image[
    detected_faces[0].rect.top():detected_faces[0].rect.bottom(),
    detected_faces[0].rect.left():detected_faces[0].rect.right(),
    :
])
# %%
faces = dlib.full_object_detections()
for detection in detected_faces:
    faces.append(shape_predictor(image, detection.rect))
# %%
fig, ax = plt.subplots(figsize=(10, 5))

ax.imshow(image)
ax.add_patch(
    mpatches.Rectangle(
        (faces[0].rect.left(), faces[0].rect.top()), faces[0].rect.width(), faces[0].rect.height(),
        edgecolor='red', fill=False,
    )
)
for point in faces[0].parts():
    ax.add_patch(
        mpatches.Circle(
            (point.x, point.y),
            edgecolor='green', fill=False,
            radius=1
        )
    )

fig.tight_layout()

# %%
