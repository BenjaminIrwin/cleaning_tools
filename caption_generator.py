import argparse
import os
import re
import sys

import torch
from PIL import Image
sys.path.insert(0,'BLIP')
from models.blip_vqa import blip_vqa
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='data/images', help='path to images')
parser.add_argument('--caption_dir', type=str, default='out')
parser.add_argument('--skip_overwrite', type=str, default=None, help='skip overwriting existing captions in --caption_dir')

class BlipDataset(Dataset):
    def __init__(self, img_dir, transform=None, skip_overwrite=None):
        self.img_dir = img_dir

        if skip_overwrite:
            already_downloaded = set(os.listdir(skip_overwrite))
            print('Skipping overwriting {} files'.format(len(skip_overwrite)))

        import glob
        types = ('*.jpg', '*.png', '*.jpeg')
        self.img_list = []
        for files in types:
            image_paths = glob.glob(img_dir + '/' + files)
            for p in image_paths:
                if already_downloaded:
                    n = re.sub('\.[^/.]+$', '', os.path.basename(p).replace('c_','t_', 1)) + '.txt'
                    if n not in already_downloaded:
                        self.img_list.append(p)
                    else:
                        print('Ignoring image {}'.format(p))
                else:
                    self.img_list.append(p)

            self.img_list.extend(image_paths)
        self.img_list.reverse()
        print('DATASET CREATED WITH ' + str(len(self.img_list)) + ' files.')
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = Image.open(self.img_list[idx]).convert('RGB')
        path = self.img_list[idx]
        if self.transform:
            image = self.transform(image)
        return image, path

    def collate_fn(self, examples):

        pixel_values = [example[0] for example in examples]
        paths = [example[1] for example in examples]

        pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()

        batch = {
            "image_pixel_values": pixel_values,
            "paths": paths
        }

        return batch


def main():
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_size = 480  # should be kept to 480 for VQA
    model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    args = parser.parse_args()

    cutout_folder = args.image_dir
    caption_folder = args.caption_dir
    ignore_images = args.ignore_images

    if not os.path.exists(caption_folder):
        os.mkdir(caption_folder)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    from torch.utils.data import DataLoader
    from pathlib import Path

    batch_size = 100

    data = BlipDataset(cutout_folder, transform, ignore_images)
    inference_dataloader = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn)

    data_iter = iter(inference_dataloader)

    question1 = 'what age group is the person?'
    question2 = 'what is the person wearing?'
    question3 = 'what is the position of the person?'
    question4 = 'what is the gender of the person?'

    i = 1
    for step, batch in enumerate(data_iter):
        print('STEP: ' + str(step))
        paths = batch['paths']
        print(paths)
        images = batch['image_pixel_values'].to(device)
        answers = {}
        with torch.no_grad():
            for q in [question1, question2, question4, question3]:
                print('QUESTION: ' + q)
                answer = model(images, q, train=False, inference='generate')
                print('ANSWERS: ')
                print(answer)
                answers[q] = answer

        i += 1

        for j in range(batch_size):
            caption = f'{answers[question1][j]} {answers[question4][j]} {answers[question3][j]} wearing {answers[question2][j]}'
            print(caption)
            key = Path(paths[j]).stem.replace('c_', 't_', 1)
            with open(f'{caption_folder}{key}.txt', "w") as text_file:
                text_file.write(caption)


if __name__ == '__main__':
    main()
