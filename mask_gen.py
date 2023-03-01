import argparse
import glob
import os
from tqdm import tqdm

from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--source_dir', type=str, default='images', help='Directory containing the source images')
parser.add_argument('--text_output_dir', type=str, default='boundings', help='txt_output_dir')
parser.add_argument('--class_name', type=str, default='person', help='Class you want to mask')
parser.add_argument('--mask_output_dir', type=str, default='masks', help='mask_output_dir')
parser.add_argument('--include_cutout', action='store_true', help='include_cutout')
parser.add_argument('--cutout_output_dir', type=str, default='cutouts', help='cutout_output_dir')


def main():
    args = parser.parse_args()
    # Create output directories
    if not os.path.exists(args.mask_output_dir):
        os.makedirs(args.mask_output_dir)
    if args.include_cutout and not os.path.exists(args.cutout_output_dir):
        os.makedirs(args.cutout_output_dir)

    # Load all the text files
    text_files = glob.glob(args.text_output_dir + '/*')
    print('Found {} text files'.format(len(text_files)))
    no_class_num = 0
    for t in tqdm(text_files):
        # Get the image name
        name = os.path.basename(t).replace('.txt', '').replace('b_', '')
        # print('Processing {}'.format(name))
        # Parse text file to get the bounding boxes
        with open(t, 'r') as f:
            lines = f.readlines()
            # Get width and height dimensions from first line
            size = tuple(map(int, lines[0].strip().split(', ')))
            try:
                # Get index of 'class_name:' lines
                class_name_idx = lines.index(args.class_name + ': \n')
                # Get index of next line that contains a ':' (i.e. the next class) or the end of the file
                next_class_idx = [i for i, line in enumerate(lines[class_name_idx + 1:]) if ':' in line]
                next_class_idx = next_class_idx[0] + class_name_idx + 1 if len(next_class_idx) > 0 else len(lines)
                # Get the bounding boxes
                mask_xyxy_list = [list(map(float, line.strip().strip('[]').split(', '))) for line in lines[class_name_idx + 1:next_class_idx]]

                for idx, mask_xyxy in enumerate(mask_xyxy_list):
                    try:
                        # Create mask
                        img = Image.new("RGB", size, (0, 0, 0))
                        d = ImageDraw.Draw(img)
                        d.rectangle(mask_xyxy, fill="white")
                        img.save(args.mask_output_dir + '/m_' + name + '_' + str(idx) + '.jpg')

                        # Create cutout
                        if args.include_cutout:
                            img_path = glob.glob(args.source_dir + '/' + name + '.*')[0]
                            img = Image.open(img_path).convert('RGB')
                            img = img.crop(mask_xyxy)
                            img.save(args.cutout_output_dir + '/c_' + name + '_' + str(idx) + '.jpg')
                    except:
                        print(f'unable to generate mask for {t}')
            except ValueError:
                # print('No {} found in {}'.format(args.class_name, t))
                no_class_num += 1
    print(f'{no_class_num} bounding files with no {args.class_name}')

if __name__ == '__main__':
    main()
