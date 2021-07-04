"""
This script trains a simple model Cats and Dogs dataset and saves it in the SavedModel format
"""
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description='Split video into frames')

    parser.add_argument('--input_dir', help='Input dir', required=True)
    parser.add_argument('--output_dir', help='Output dir', required=True)
    parser.add_argument('--output_filename', help='Output file name', required=True)
    parser.add_argument('--input_name_template', default='frame_%5d.png', help='Input name template', required=False)
    parser.add_argument('--fps', default=30, type=int, help='Frames per second', required=False)

    args = parser.parse_args()

    return args


def main():
    # TODO: fmpeg -i face_close_aged_long.mp4 -filter "minterpolate='fps=120'" zzzz3.mp4
    # https://superuser.com/a/1185430
    # https://github.com/dthpham/butterflow

    args = parse_args()
    print(args)

    input_filename = f'{args.input_dir}/{args.input_name_template}'
    output_filename = f'{args.output_dir}/{args.output_filename}'
    cmd = [
        'ffmpeg', '-y', '-framerate', str(args.fps), '-i', input_filename,
        '-c:v', 'libx264', '-vf', f'fps={args.fps},pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
        output_filename
    ]
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
