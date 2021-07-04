import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description='Split video into frames')

    parser.add_argument('--input_dir', help='Input dir', required=True)
    parser.add_argument('--input_filename', help='Input file name', required=True)
    parser.add_argument('--output_dir', help='Output dir', required=True)
    parser.add_argument('--output_name_template', default='frame_%5d.png', help='Output name template', required=False)
    parser.add_argument('--fps', default=30, type=int, help='Frames per second', required=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print(args)

    input_filename = f'{args.input_dir}/{args.input_filename}'
    output_filename = f'{args.output_dir}/{args.output_name_template}'
    cmd = [
        'ffmpeg', '-i', input_filename, '-vf', f'fps={args.fps}', output_filename
    ]
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
