import os
import sys
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.cuda
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Split video into frames')

    parser.add_argument('--input_dir', help='Input dir', required=True)
    parser.add_argument('--output_dir', help='Output dir', required=True)
    parser.add_argument('--aging_dir', default='/svr/code/aging', help='Aging dir', required=False)
    parser.add_argument('--config', default='001', help='Aging config', required=False)
    parser.add_argument('--age_start', default=21, help='Start age', required=False)
    parser.add_argument('--age_end', default=69, help='End age', required=False)
    parser.add_argument('--delay', default=0.3, help='Start and end delay', required=False)

    args = parser.parse_args()

    return args


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def create_trainer(args):
    from trainer import Trainer

    config_path = os.path.join(args.aging_dir, 'configs', f'{args.config}.yaml')
    with open(config_path, 'r') as file_stream:
        config = yaml.load(file_stream)
    print(f'Config loaded: {config}')

    trainer = Trainer(config)

    checkpoint_filename = os.path.join(args.aging_dir, 'logs', args.config, 'checkpoint')
    trainer.load_checkpoint(checkpoint_filename)
    trainer.to(get_device())
    print(f'Trainer created: {type(trainer)}, {checkpoint_filename}')

    return trainer


def preprocess(image, config):
    image_size = (config['input_w'], config['input_h'])

    resize = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    normalize = transforms.Normalize(mean=[0.48501961, 0.45795686, 0.40760392], std=[1, 1, 1])

    image = resize(image)
    if image.size(0) == 1:
        image = torch.cat((image, image, image), dim=0)
    image = normalize(image)

    return image


def load_image(filename):
    image = Image.open(filename)

    return image


def process_images(trainer, images, age_start, age_end, delay):
    from functions import clip_img

    nb_images = len(images)
    nb_delay = int(nb_images * delay)
    nb_transition = nb_images - nb_delay * 2

    transition_ages = np.linspace(age_start, age_end, num=nb_transition, dtype=np.long).tolist()
    images_ages = [age_start, ] * nb_delay + transition_ages + [age_end, ] * nb_delay

    print(f'Ages: {len(images_ages)}')

    device = get_device()
    images_aged = []
    for image_filename, target_age in zip(tqdm(images, desc='Aging'), images_ages):
        with torch.no_grad():
            image = load_image(image_filename)
            image_size = image.size

            image = preprocess(image, trainer.config)
            image = image.unsqueeze(0).to(device)
            target_age_tensor = torch.tensor(target_age).unsqueeze(0).to(device)

            image_aged = trainer.test_eval(image, target_age_tensor, target_age=target_age, hist_trans=True)

            image_aged = clip_img(image_aged)[0]
            image_aged = image_aged.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

            image_aged = Image.fromarray(image_aged)
            image_aged = image_aged.resize(image_size)

            images_aged.append(image_aged)

    return images_aged


def main():
    args = parse_args()
    print(args)

    # append aging to path
    sys.path.insert(1, args.aging_dir)

    # create trainer
    trainer = create_trainer(args)

    # get images files
    images = sorted(Path(args.input_dir).glob('*.png'))
    print(f'Images: {len(images)}')

    # age images
    images_aged = process_images(trainer, images, args.age_start, args.age_end, args.delay)
    print(f'Images aged: {len(images_aged)}')

    # save images
    output_dir = Path(args.output_dir)
    for image_filename, image_aged in zip(images, images_aged):
        image_aged.save(output_dir.joinpath(image_filename.name))


if __name__ == '__main__':
    main()
