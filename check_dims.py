import argparse
import glob
import os
import random
import re

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--source_dir', type=str, default='images', help='Directory containing the source images')
parser.add_argument('--mask_dir', type=str, default='masks', help='Directory containing the source masks')
parser.add_argument('--dry_run', action='store_true', help='will not delete files if true')


def main():
    args = parser.parse_args()
    source_dir = args.source_dir
    dry_run = args.dry_run

    print('Checking images in', source_dir)

    violating_ims = []

    # Iterate through all files in the source directory
    for file in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file)
        try:
            # Try to open the file
            img = Image.open(file_path)
            if img.size != (512, 512):
                violating_ims.append(file_path)
        except (IOError, SyntaxError) as e:
            print('Bad file:', file_path)

    print('Found', len(violating_ims), 'violating images')
    print('Would you like to crop {} files?'.format(len(violating_ims)))
    print('This cannot be undone.')
    answer = input('y/n: ')
    if answer == 'y':
        for img_path in violating_ims:
            # Open image
            img = Image.open(img_path)
            print('Image size:', img.size)
            # Delete image if one dim is smaller than 512
            if img.size[0] < 512 or img.size[1] < 512:
                print('Deleting image', img_path)
                if not dry_run:
                    os.remove(img_path)
                continue
            # Open all corresponding masks
            basename = re.sub('\.[^/.]+$', '', os.path.basename(img_path))
            masks = glob.glob(os.path.join(args.mask_dir, 'm_' + basename + '*'))
            print('Found', len(masks), 'masks for', img_path)
            # Perform same random crop on image and masks
            x = random.randint(0, img.size[0] - 512)
            # x = 0
            # y = random.randint(0, img.size[1] - 512)
            y = img.size[1] - 512
            img = img.crop((x, y, x + 512, y + 512))

            if dry_run:
                img.show()
            else:
                img.save(img_path)

            for mask in masks:
                m = Image.open(mask).convert('L')
                m = m.crop((x, y, x + 512, y + 512))
                # Only save mask if it contains at least one white pixel
                if np.any(np.array(m) == 255):
                    print('Saving mask', mask)
                    if dry_run:
                        m.show()
                    else:
                        m.save(mask)
                else:
                    print('Deleting mask', mask)
                    if not dry_run:
                        os.remove(mask)
    else:
        print('Aborted')


if __name__ == '__main__':
    main()
