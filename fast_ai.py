import argparse
from fastcore.all import *
from fastai.vision.all import *
import os
import glob
import shutil
import torch

from util import get_batch

parser = argparse.ArgumentParser(description='Fast AI dataset cleaner')
parser.add_argument('--input_image_folder', type=str, required=True,
                    help='The folder containing the images to be classified.')
parser.add_argument('--classifier_path', type=str, default='render_classifier (1).pkl',
                    help='Path to the classifier model.')
parser.add_argument('--batch_size', type=int, default=50, help='Inference batch size.')
parser.add_argument('--remove_if_false', type=bool, default=False, help='Remove images if classifier returns false.')

trash_folder = 'trash'
failed_folder = 'failed'

if not os.path.exists(trash_folder):
    os.makedirs(trash_folder)

if not os.path.exists(failed_folder):
    os.makedirs(failed_folder)


def main():
    print("***** FAST AI CLASSIFIER *****")

    args = parser.parse_args()
    custom_classifier_path = args.classifier_path
    input_image_folder = args.input_image_folder
    batch_size = args.batch_size
    remove_if_false = args.remove_if_false
    learn = load_learner(custom_classifier_path, cpu=False)
    learn.model.to(torch.device('cuda'))

    a = 0
    test_files = []
    while len(test_files) == 0 and a < 10:
        test_files = glob.glob(input_image_folder + '/*.jpg')

    print('FILES FOUND: ', len(test_files))
    num_batches = len(test_files) / batch_size
    print('BATCHES: ', num_batches)
    curr_batch = 1
    for batch in get_batch(test_files, batch_size):
        test_dl = learn.dls.test_dl(batch)
        print('BATCH: ', curr_batch, ' OF ', num_batches)
        curr_batch += 1
        try:
            preds, _, decoded = learn.get_preds(dl=test_dl, with_decoded=True)
            count = 0
            for idx, pred in enumerate(preds):
                if pred[0] < 0.85:
                    count += 1
                    filename = os.path.basename(batch[idx])
                    shutil.move(batch[idx], trash_folder + '/' + filename)

            print('CLEANED: ', count)
        except Exception:
            print('BATCH ERROR THEREFORE TRYING IMAGES INDIVIDUALLY')
            count = 0
            for image in batch:
                try:
                    outcome, _, probs = learn.predict(PILImage.create(image))
                    count = 0
                    if probs[0] < 0.85:
                        count += 1
                        filename = os.path.basename(image)
                        shutil.move(image, trash_folder + '/' + filename)

                    print('CLEANED: ', count)
                except Exception:
                    print('ERROR MOVING IMAGE TO FAILED.')
                    filename = os.path.basename(image)
                    shutil.move(image, failed_folder + '/' + filename)

    cleaned_images = glob.glob(trash_folder + '/*')
    print('CLEANED IMAGES: ', len(cleaned_images))


if __name__ == "__main__":
    main()
