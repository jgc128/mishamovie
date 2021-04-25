FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.0-cudnn7-ubuntu18.04

RUN apt-get update && apt install -y libgl1-mesa-dev ffmpeg git cmake

# download dblib models
RUN mkdir -p /srv/models/
RUN cd /srv/models/ && \
    curl -Lo shape_predictor_5_face_landmarks.dat.bz2 https://mishamoviestorage.blob.core.windows.net/pub/dlib/shape_predictor_5_face_landmarks.dat.bz2 && \ 
    bzip2 -d shape_predictor_5_face_landmarks.dat.bz2 && \
    curl -Lo shape_predictor_68_face_landmarks.dat.bz2 https://mishamoviestorage.blob.core.windows.net/pub/dlib/shape_predictor_68_face_landmarks.dat.bz2 && \ 
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# install the requirements
COPY requirements.txt /srv/
RUN pip install -r /srv/requirements.txt
