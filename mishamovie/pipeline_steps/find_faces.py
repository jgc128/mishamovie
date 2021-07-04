import os
import argparse
from pathlib import Path

import dlib
from tqdm import tqdm

from mishamovie.utils.io import save_json


def parse_args():
    parser = argparse.ArgumentParser(description='Split video into frames')

    parser.add_argument('--input_dir', help='Input dir', required=True)
    parser.add_argument('--output_dir', help='Output dir', required=True)
    parser.add_argument('--dlib_models_dir', help='dlib models dir', required=True)
    parser.add_argument(
        '--face_detector_filename', default='mmod_human_face_detector.dat',
        help='dlib face detector filename', required=False
    )
    parser.add_argument(
        '--shape_predictor_filename', default='shape_predictor_68_face_landmarks_GTX.dat',
        help='dlib shape predictor path', required=False
    )
    parser.add_argument('--output_filename', default='faces.json', help='Output file name', required=False)

    args = parser.parse_args()

    return args


def load_dlib(args):
    face_detector_path = os.path.join(args.dlib_models_dir, args.face_detector_filename)
    shape_predictor_path = os.path.join(args.dlib_models_dir, args.shape_predictor_filename)

    face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)
    shape_predictor = dlib.shape_predictor(shape_predictor_path)

    return face_detector, shape_predictor


def dlib_to_json(full_object_detection):
    result = {
        'rect': {
            'left': full_object_detection.rect.left(),
            'top': full_object_detection.rect.top(),
            'width': full_object_detection.rect.width(),
            'height': full_object_detection.rect.height(),
        },
        'points': [
            {'x': p.x, 'y': p.y}
            for p in full_object_detection.parts()
        ],
    }

    return result


def find_face_landmarks(face_detector, shape_predictor, image):
    detected_faces = face_detector(image, 1)
    detected_face = detected_faces[0]

    full_object_detection = shape_predictor(image, detected_face.rect)
    face_landmarks = dlib_to_json(full_object_detection)

    return face_landmarks


def main():
    args = parse_args()
    print(args)

    face_detector, shape_predictor = load_dlib(args)
    print(f'DLib loaded: {face_detector}, {shape_predictor}')

    images = sorted(Path(args.input_dir).glob('*.png'))
    faces = {}
    for image_filename in tqdm(images, desc='Processing'):
        # image.shape (540, 1024, 3)
        image = dlib.load_rgb_image(str(image_filename))
        face = find_face_landmarks(face_detector, shape_predictor, image)

        faces[image_filename.name] = face

    # save faces info
    output_filename = os.path.join(args.output_dir, args.output_filename)
    save_json(faces, output_filename)

    print(f'Saved: {output_filename}')


if __name__ == '__main__':
    main()
