import argparse
import glob
import os
import shutil

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from torchvision import transforms as T

from util import get_batch

preprocessing = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

trash_folder = 'trash'
failed_folder = 'failed'

parser = argparse.ArgumentParser(description='Watermark cleaner')
parser.add_argument('--input_image_folder', type=str, required=True,
                    help='The folder containing the images to be classified.')
parser.add_argument('--batch_size', type=int, default=50, help='Inference batch size.')


def detect_watermark(image_paths, conf_thresh=0.5):
    model = timm.create_model(
        'efficientnet_b3a', pretrained=True, num_classes=2)

    model.classifier = nn.Sequential(
        # 1536 is the orginal in_features
        nn.Linear(in_features=1536, out_features=625),
        nn.ReLU(),  # ReLu to be the activation function
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=2),
    )

    state_dict = torch.load('watermark_model_v1.pt')

    model.load_state_dict(state_dict)
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    watermark_images = []

    img_batch = []
    for image_path in image_paths:
        img_batch.append(preprocessing(Image.open(image_path).convert('RGB')))
    batch = torch.stack(img_batch).cuda()
    with torch.no_grad():
        pred = model(batch)
        syms = F.softmax(pred, dim=1).detach().cpu().numpy().tolist()
        for idx, water_sym in enumerate(syms):
            if water_sym[0] > conf_thresh:
                watermark_images.append(image_paths[idx])

    return watermark_images


def main():
    print("***** WATERMARK DETECTOR *****")

    args = parser.parse_args()
    input_image_folder = args.input_image_folder
    batch_size = args.batch_size
    test_files = glob.glob(input_image_folder + '/*.jpg')
    print('FILES FOUND: ', len(test_files))
    num_batches = len(test_files) / batch_size
    print('BATCHES: ', num_batches)
    curr_batch = 1
    for batch in get_batch(test_files, batch_size):
        print('BATCH: ', curr_batch, ' OF ', num_batches)
        try:
            watermark_images = detect_watermark(batch)
            # Move all watermark images to 'failed' folder
            for img in watermark_images:
                print('Moving ' + img)
                filename = os.path.basename(img)
                shutil.move(img, trash_folder + '/' + filename)
        except Exception as e:
            print('ERROR: ', e)
        curr_batch += 1


if __name__ == "__main__":
    main()
