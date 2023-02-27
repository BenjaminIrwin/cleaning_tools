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
parser.add_argument('--working_dir', type=str, default='.', help='path where the images to ignore will be saved')
parser.add_argument('--class_name', type=str, default='person', help='Class you want to generate caption for')
parser.add_argument('--batch_size', type=int, default=100, help='batch_size')


class BlipDataset(Dataset):
    def __init__(self, img_dir, transform=None, ignore_images=None):
        self.img_dir = img_dir

        import glob
        types = ('*.jpg', '*.png', '*.jpeg')
        self.img_list = []
        for files in types:
            image_paths = glob.glob(img_dir + '/' + files)
            for p in image_paths:
                self.img_list.append(p)

            self.img_list.extend(image_paths)
            self.img_list = list(set(self.img_list))
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
    working_dir = args.working_dir
    batch_size = args.batch_size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    from torch.utils.data import DataLoader
    from pathlib import Path

    data = BlipDataset(cutout_folder, transform)
    inference_dataloader = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn)

    data_iter = iter(inference_dataloader)

    if class_name == 'person':
        QA_question = 'is this a person?'
    
    elif class_name == 'car':
        QA_question = 'is it a car?'

    tot_no_class = 0
    images_with_no_class = []
    for step, batch in enumerate(data_iter):
        print('\n STEP: ' + str(step+1) + ' of ' + str(len(data_iter)))
        paths = batch['paths']
        # print(paths)
        images = batch['image_pixel_values'].to(device)
        answers = {}
        with torch.no_grad():
            QA_check = model(images, QA_question, train=False, inference='generate')
        
        stp_no_class = 0
        for j in range(batch_size):
            try:
                if QA_check[j] == 'no':
                    key = Path(paths[j]).stem.replace('c_', '', 1)
                    images_with_no_class += [key]
                    tot_no_class+=1
                    stp_no_class+=1
            
            except:
                    # this shouldn't be saved here but I couldn't be bother fixing the error that always comes up at the end
                    print(f'{i} images with no classes') 
                    print(f'images_with_no_class.txt saved in {working_dir}')
                    images_with_no_class = list(set(images_with_no_class))
                    with open(working_dir + class_name + '_images_with_no_class.txt', 'w') as f:
                        for line in images_with_no_class:
                            f.write(f"{line}\n")
                    exit()
                    
        print(f'{stp_no_class} images with no class in step')

    print(f'images_with_no_class.txt saved in {working_dir}')
    print(f'{tot_no_class} images with no classes') 
    images_with_no_class = list(set(images_with_no_class))
    with open(working_dir + class_name + '_images_with_no_class.txt', 'w') as f:
        for line in images_with_no_class:
            f.write(f"{line}\n")


if __name__ == '__main__':
    main()
