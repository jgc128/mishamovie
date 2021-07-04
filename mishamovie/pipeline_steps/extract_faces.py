import os
import argparse
from pathlib import Path
import dataclasses


import dlib
from tqdm import tqdm

from mishamovie.utils.io import load_json, save_json


@dataclasses.dataclass
class FaceBoxInfo:
    buffer_x: int
    buffer_y: int
    x1: int
    x2: int
    y1: int
    y2: int


def parse_args():
    parser = argparse.ArgumentParser(description='Extract faces from frames')

    parser.add_argument('--input_dir', help='Input dir', required=True)
    parser.add_argument('--output_dir', help='Output dir', required=True)
    parser.add_argument('--faces_info_dir', help='Faces info dir', required=True)
    parser.add_argument('--faces_info_filename', default='faces.json', help='Faces info file name', required=False)
    parser.add_argument(
        '--faceboxes_info_filename', default='faceboxes_info.json', help='Faces boxes info file name', required=False
    )
    parser.add_argument('--buffer', default=0.3, help='Margin around face', required=False)

    args = parser.parse_args()

    return args


def extract_face(image, face_info, buffer):
    face_x1 = face_info['rect']['left']
    face_x2 = face_info['rect']['left'] + face_info['rect']['width']
    face_y1 = face_info['rect']['top']
    face_y2 = face_info['rect']['top'] + face_info['rect']['height']

    buffer_x = int((face_x2 - face_x1) * buffer)
    buffer_y = int((face_y2 - face_y1) * buffer)

    facebox_x1 = face_x1 - buffer_x
    facebox_x2 = face_x2 + buffer_x
    facebox_y1 = face_y1 - buffer_y
    facebox_y2 = face_y2 + buffer_y

    facebox_info = FaceBoxInfo(
        buffer_x, buffer_y, facebox_x1, facebox_x2, facebox_y1, facebox_y2
    )

    facebox = image[facebox_info.y1:facebox_info.y2, facebox_info.x1:facebox_info.x2, :]

    return facebox, facebox_info


def main():
    args = parse_args()
    print(args)

    faces_info_filename = os.path.join(args.faces_info_dir, args.faces_info_filename)
    faces_info = load_json(faces_info_filename)
    print(f'Faces info loaded: {len(faces_info)}')

    images = sorted(Path(args.input_dir).glob('*.png'))
    output_dir = Path(args.output_dir)
    faceboxes_info = {}
    for image_filename in tqdm(images, desc='Processing'):
        # image.shape (540, 1024, 3)
        image = dlib.load_rgb_image(str(image_filename))
        face_info = faces_info[image_filename.name]

        facebox, facebox_info = extract_face(image, face_info, args.buffer)

        facebox_filename = output_dir.joinpath(image_filename.name)
        dlib.save_image(facebox, str(facebox_filename))

        # update buffer info
        faceboxes_info[image_filename.name] = dataclasses.asdict(facebox_info)

    # save buffer info
    faceboxes_info_filename = os.path.join(args.faces_info_dir, args.faceboxes_info_filename)
    save_json(faceboxes_info, faceboxes_info_filename)
    print(f'Buffer info saved: {faceboxes_info_filename}')


if __name__ == '__main__':
    main()
