import os
import json
import argparse
import subprocess
from pathlib import Path

import dlib
from tqdm import tqdm

from mishamovie.utils.io import load_json, save_json


def parse_args():
    parser = argparse.ArgumentParser(description='Extract faces from frames')

    parser.add_argument('--input_dir', help='Input dir', required=True)
    parser.add_argument('--output_dir', help='Output dir', required=True)
    parser.add_argument('--faces_info_dir', help='Faces info dir', required=True)
    parser.add_argument('--faces_info_filename', default='faces.json', help='Faces info file name', required=False)
    parser.add_argument('--buffer_info_filename', default='buffer.json', help='Buffer info file name', required=False)
    parser.add_argument('--buffer', default=0.3, help='Margin around face', required=False)

    args = parser.parse_args()

    return args


def extract_face(image, face_info, buffer):
    x1 = face_info['rect']['top']
    x2 = face_info['rect']['top'] + face_info['rect']['height']

    y1 = face_info['rect']['left']
    y2 = face_info['rect']['left'] + face_info['rect']['width']

    buffer_x = int((x2 - x1) * buffer)
    buffer_y = int((y2 - y1) * buffer)

    face = image[x1 - buffer_x:x2 + buffer_x, y1 - buffer_y:y2 + buffer_y, :]

    return face, buffer_x, buffer_y


def main():
    args = parse_args()
    print(args)

    faces_info_filename = os.path.join(args.faces_info_dir, args.faces_info_filename)
    faces_info = load_json(faces_info_filename)
    print(f'Faces info loaded: {len(faces_info)}')

    images = sorted(Path(args.input_dir).glob('*.png'))
    output_dir = Path(args.output_dir)
    buffer_info = {}
    for image_filename in tqdm(images, desc='Processing'):
        # image.shape (540, 1024, 3)
        image = dlib.load_rgb_image(str(image_filename))
        face_info = faces_info[image_filename.name]

        face, buffer_x, buffer_y = extract_face(image, face_info, args.buffer)

        face_filename = output_dir.joinpath(image_filename.name)
        dlib.save_image(face, str(face_filename))

        # update buffer info
        buffer_info[image_filename.name] = {'x': buffer_x, 'y': buffer_y}

    # save buffer info
    buffer_info_filename = os.path.join(args.faces_info_dir, args.buffer_info_filename)
    save_json(buffer_info, buffer_info_filename)
    print(f'Buffer info saved: {buffer_info_filename}')


if __name__ == '__main__':
    main()
