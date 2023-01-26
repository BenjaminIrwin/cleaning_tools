import argparse
import os
import glob

parser = argparse.ArgumentParser(description='Dataset cleaner')
parser.add_argument('--base_path', type=str, required=True)
parser.add_argument('--output_folder', type=str, required=True)

def main():
    args = parser.parse_args()

    base_path = args.base_path
    output_folder = args.output_folder

    a = 0
    mask_files = []
    while len(mask_files) == 0 and a < 10:
      mask_files = glob.glob(base_path + '/masks/*')

    im_names = []

    for m in mask_files:
      f_name = os.path.basename(m).replace('m', '').replace('.jpg', '')
      im_name = base_path + '/original/i' + f_name + '.jpg'
      im_names.append(im_name)

    print('IMAGES FOUND: ', len(im_names))
    print('EXAMPLES: ', im_names[:10])
    # Asl the user if they want to move the images
    move = input('Move images to ' + output_folder + '? (y/n): ')
    if move == 'y':
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for im in im_names:
            f_name = os.path.basename(im)
            os.rename(im, output_folder + '/' + f_name)
            print('MOVED: ', f_name)

