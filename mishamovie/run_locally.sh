#!/usr/bin/env bash

FPS=1
SOURCE_FILENAME=face_standing.mp4

DATA_DIR=/home/azureuser/cloudfiles/code/data
DLIB_MODELS_DIR=/home/azureuser/cloudfiles/code/Users/jgc128/dlib-models
SOURCE_DIR=${DATA_DIR}/source
FRAMES_DIR=${DATA_DIR}/frames
FACES_INFO_DIR=${DATA_DIR}/faces_info
FACES_DIR=${DATA_DIR}/faces

# # Split to frames
# python pipeline_steps/split_to_frames.py \
#     --input_dir ${SOURCE_DIR} \
#     --input_filename ${SOURCE_FILENAME} \
#     --output_dir ${FRAMES_DIR} \
#     --fps ${FPS}

# # Find faces on the frames
# python pipeline_steps/find_faces.py \
#     --input_dir ${FRAMES_DIR} \
#     --output_dir ${FACES_INFO_DIR} \
#     --dlib_models_dir ${DLIB_MODELS_DIR}

# Extract faces from the frames
python pipeline_steps/extract_faces.py \
    --input_dir ${FRAMES_DIR} \
    --output_dir ${FACES_DIR} \
    --faces_info_dir ${FACES_INFO_DIR}
