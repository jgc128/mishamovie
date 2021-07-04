import os
import argparse
from pathlib import Path

import cv2
import dlib
import numpy as np
from tqdm import tqdm

from mishamovie.utils.io import load_json


def parse_args():
    parser = argparse.ArgumentParser(description='Split video into frames')

    parser.add_argument('--frames_dir', help='Input dir with frames', required=True)
    parser.add_argument('--faces_dir', help='Input dir with faces', required=True)
    parser.add_argument('--output_dir', help='Output dir', required=True)
    parser.add_argument('--faces_info_dir', help='Faces info dir', required=True)
    parser.add_argument('--faces_info_filename', default='faces.json', help='Faces info file name', required=False)
    parser.add_argument(
        '--faceboxes_info_filename', default='faceboxes_info.json', help='Faces boxes info file name', required=False
    )

    args = parser.parse_args()

    return args


def insert_face(frame, face, face_info, facebox_info):
    # 1: Just insert
    # frame[facebox_info['y1']:facebox_info['y2'], facebox_info['x1']:facebox_info['x2'], :] = face

    # 2: seamlessClone without mask
    # face_mask = np.full_like(face, 255)
    # face_center_coord = (
    #     facebox_info['x1'] + (facebox_info['x2'] - facebox_info['x1']) // 2,
    #     facebox_info['y1'] + (facebox_info['y2'] - facebox_info['y1']) // 2,
    # )

    # frame_with_face = cv2.seamlessClone(face, frame, face_mask, face_center_coord, cv2.NORMAL_CLONE)

    # 3: seamlessClone with a mask

    face_landmarks_array = np.array([
        [p['x'] - facebox_info['x1'], p['y'] - facebox_info['y1']]
        for p in face_info['points']
    ])
    face_hull = cv2.convexHull(face_landmarks_array, returnPoints=True)[:, 0, :]

    face_mask = np.full_like(face, 0)
    cv2.fillConvexPoly(face_mask, face_hull, color=(255, 255, 255))

    kernel_size = face_info['rect']['width'] // 10
    face_mask_dilated = cv2.dilate(face_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

    contours, _ = cv2.findContours(face_mask_dilated[:, :, 0] * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = np.concatenate(contours, axis=0)[:, 0, :]
    br_x, br_y, br_w, br_h = cv2.boundingRect(contours)
    center = (
        facebox_info['x1'] + br_x + br_w // 2,
        facebox_info['y1'] + br_y + br_h // 2,
    )
    frame_with_face = cv2.seamlessClone(face, frame, face_mask_dilated, center, cv2.NORMAL_CLONE)

    return frame_with_face


def main():
    args = parse_args()
    print(args)

    faces_info_filename = os.path.join(args.faces_info_dir, args.faces_info_filename)
    faces_info = load_json(faces_info_filename)

    faceboxes_info_filename = os.path.join(args.faces_info_dir, args.faceboxes_info_filename)
    faceboxes_info = load_json(faceboxes_info_filename)

    frames = {image.name: image for image in Path(args.frames_dir).glob('*.png')}
    faces = {image.name: image for image in Path(args.faces_dir).glob('*.png')}
    output_dir = Path(args.output_dir)
    for image_filename in tqdm(frames.keys(), desc='Processing'):
        frame_filename = frames[image_filename]
        face_filename = faces[image_filename]

        # image.shape (540, 1024, 3)
        frame = dlib.load_rgb_image(str(frame_filename))
        face = dlib.load_rgb_image(str(face_filename))

        facebox_info = faceboxes_info[image_filename]
        face_info = faces_info[image_filename]

        frame = insert_face(frame, face, face_info, facebox_info)

        frames_inserted_filename = output_dir.joinpath(image_filename)
        dlib.save_image(frame, str(frames_inserted_filename))


if __name__ == '__main__':
    main()
