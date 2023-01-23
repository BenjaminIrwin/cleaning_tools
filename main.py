import argparse
from fastcore.all import *
from fastai.vision.all import *
import os
import glob
import shutil
import torch

parser = argparse.ArgumentParser(description='Fast AI dataset cleaner')
parser.add_argument('--input_image_folder', type=str, required=True,
                    help='The folder containing the images to be classified.')
parser.add_argument('--classifier_path', type=str, default='render_classifier (1).pkl',
                    help='Path to the classifier model.')
parser.add_argument('--batch_size', type=int, default=50, help='Inference batch size.')

trash_folder = 'trash'
failed_folder = 'failed'

if not os.path.exists(trash_folder):
    os.makedirs(trash_folder)

if not os.path.exists(failed_folder):
    os.makedirs(failed_folder)


def get_batch(list, batch_size):
    # Iterate over a list with a batch size
    for i in range(0, len(list), batch_size):
        yield list[i:i + batch_size]


def main():
    print("***** FAST AI CLASSIFIER *****")

    args = parser.parse_args()
    custom_classifier_path = args.classifier_path
    input_image_folder = args.input_image_folder
    batch_size = args.batch_size
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
            try:
                for img in batch:
                    if os.path.exists(img):
                        filename = os.path.basename(img)
                        shutil.move(img, failed_folder + '/' + filename)
            except Exception:
                print('ERROR MOVING FILES TO FAILED.')

        curr_batch += 1

    cleaned_images = glob.glob(trash_folder + '/*.png')
    print('CLEANED IMAGES: ', len(cleaned_images))


if __name__ == "__main__":
    main()
