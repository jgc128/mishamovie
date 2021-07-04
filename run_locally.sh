#!/usr/bin/env bash

FPS=30
SOURCE_FILENAME=face_close_long.mov
RESULT_FILENAME=face_close_aged_long.mp4

DATA_DIR=/home/azureuser/cloudfiles/code/data
DLIB_MODELS_DIR=/home/azureuser/cloudfiles/code/Users/jgc128/dlib-models
SOURCE_DIR=${DATA_DIR}/source
FRAMES_DIR=${DATA_DIR}/frames
FACES_INFO_DIR=${DATA_DIR}/faces_info
FACES_DIR=${DATA_DIR}/faces
FACES_AGED_DIR=${DATA_DIR}/faces_aged
FRAMES_AGED_DIR=${DATA_DIR}/frames_aged
TMP_DIR=${DATA_DIR}/tmp

# # Split to frames
# python mishamovie/pipeline_steps/split_to_frames.py \
#     --input_dir ${SOURCE_DIR} \
#     --input_filename ${SOURCE_FILENAME} \
#     --output_dir ${FRAMES_DIR} \
#     --fps ${FPS}

# # Find faces on the frames
# python mishamovie/pipeline_steps/find_faces.py \
#     --input_dir ${FRAMES_DIR} \
#     --output_dir ${FACES_INFO_DIR} \
#     --dlib_models_dir ${DLIB_MODELS_DIR}

# # Extract faces from the frames
# python mishamovie/pipeline_steps/extract_faces.py \
#     --input_dir ${FRAMES_DIR} \
#     --output_dir ${FACES_DIR} \
#     --faces_info_dir ${FACES_INFO_DIR}

# Insert aged faces into the frames
python mishamovie/pipeline_steps/insert_faces.py \
    --frames_dir ${FRAMES_DIR} \
    --faces_dir ${FACES_AGED_DIR} \
    --output_dir ${FRAMES_AGED_DIR} \
    --faces_info_dir ${FACES_INFO_DIR}

# Combine frames
python mishamovie/pipeline_steps/combine_frames.py \
    --input_dir ${FRAMES_AGED_DIR} \
    --output_dir ${TMP_DIR} \
    --output_filename ${RESULT_FILENAME} \
    --fps ${FPS}
