import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, 'BLIP')
from models.blip_vqa import blip_vqa
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='data/images', help='path to images')
parser.add_argument('--caption_dir', type=str, default='out')
parser.add_argument('--ignore_images', type=str, default=None, help='path to txt file with images to ignore')
parser.add_argument('--working_dir', type=str, default='.', help='path to where the txt file with the result summary will be saved')
parser.add_argument('--class_name', type=str, default='person', help='Class you want to generate caption for')
parser.add_argument('--batch_size', type=int, default=100, help='batch_size')


class BlipDataset(Dataset):
    def __init__(self, img_dir, transform=None, ignore_images=None):
        self.img_dir = img_dir

        if ignore_images:
            with open(ignore_images, 'r') as f:
                ignore_images = f.read().splitlines()

            print('Ignoring images: {}'.format(len(ignore_images)))
            print(ignore_images)

        import glob
        types = ('*.jpg', '*.png', '*.jpeg')
        self.img_list = []
        for files in types:
            image_paths = glob.glob(img_dir + '/' + files)
            for p in image_paths:
                if ignore_images:
                    n = os.path.basename(p).replace('c_', '', 1).replace('.png', '').replace('.jpg', '').replace(
                        '.jpeg', '')
                    print('Checking image {}'.format(n))
                    # print('Checking image {}'.format(n))
                    if n not in ignore_images:
                        self.img_list.append(p)
                    else:
                        print('Ignoring image {}'.format(p))
                else:
                    self.img_list.append(p)

            self.img_list.extend(image_paths)
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

    class_name = args.class_name
    cutout_folder = args.image_dir
    caption_folder = args.caption_dir
    working_dir = args.working_dir
    batch_size = args.batch_size
    
    if not os.path.exists(caption_folder):
        os.mkdir(caption_folder)
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

    data = BlipDataset(cutout_folder, transform, ignore_images)
    inference_dataloader = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn)

    data_iter = iter(inference_dataloader)

    if class_name == 'person':
        question0 = 'is this a person?'
        question1 = 'what age group is the person?'
        question2 = 'what is the person wearing?'
        question3 = 'what is the position of the person?'
        question4 = 'what is the gender of the person?'
        question5 = 'is the image a close-up?'

        questions = [question0, question1, question2, question4, question3, question5]
    
    elif class_name == 'car':
        question0 = 'is it a car?'
        question1 = 'what is the color of the car?'
        question2 = 'what type of car is it?'
        question3 = 'is the car parked?'
        question4 = 'is the image a close-up?'

        questions = [question0, question1, question2, question3, question4]


    i = 1
    images_with_no_class = []
    for step, batch in enumerate(data_iter):
        print('\n STEP: ' + str(step+1) + ' of ' + str(len(data_iter)))
        paths = batch['paths']
        # print(paths)
        images = batch['image_pixel_values'].to(device)
        answers = {}
        with torch.no_grad():
            for q in questions:
#                 print('QUESTION: ' + q)
                answer = model(images, q, train=False, inference='generate')
#                 print('ANSWERS: ')
#                 print(answer)
                answers[q] = answer

        i += 1
        for j in range(batch_size):
            try:
                if class_name == 'person':
                    if answers[question0][j] == 'no':
                        caption = ''
                    else:
                        caption = f'{answers[question1][j]} {answers[question4][j]} {answers[question3][j]} wearing {answers[question2][j]}.'
                        if answers[question5][j] == 'yes':
                            caption = 'close-up of ' + caption

                elif class_name == 'car':
                        if answers[question0][j] == 'no':
                            caption = ''
                        else:
                            if answers[question3][j] == 'yes':
                                caption = f'{answers[question1][j]} parked {answers[question2][j]}'
                            else:
                                caption = f'{answers[question1][j]} {answers[question2][j]}'

                            if answers[question4][j] == 'yes':
                                caption = 'close-up of ' + caption

            except:
                    # this shouldn't be saved here but I couldn't be bother fixing the error that always comes up at the end
                    print(f'images_with_no_class.txt saved in {working_dir}')
                    images_with_no_class = list(set(images_with_no_class))
                    with open(working_dir + class_name + '_images_with_no_class.txt', 'w') as f:
                        for line in images_with_no_class:
                            f.write(f"{line}\n")

            print(caption)
            
            key = Path(paths[j]).stem.replace('c_', 't_', 1)
            if caption == '':
                images_with_no_class += [key]

            with open(f'{caption_folder}{key}.txt', "w") as text_file:
                text_file.write(caption)

    print(f'images_with_no_class.txt saved in {working_dir}')
    images_with_no_class = list(set(images_with_no_class))
    with open(working_dir + class_name + '_images_with_no_class.txt', 'w') as f:
        for line in images_with_no_class:
            f.write(f"{line}\n")


if __name__ == '__main__':
    main()
