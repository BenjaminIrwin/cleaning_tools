import argparse
import glob
import os

from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--source_dir', type=str, default='images', help='Directory containing the source images')
parser.add_argument('--text_output_dir', type=str, default='boundings', help='txt_output_dir')
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
    for t in text_files:
        # Get the image name
        name = os.path.basename(t).replace('.txt', '')
        print('Processing {}'.format(name))
        # Parse text file to get the bounding boxes
        with open(t, 'r') as f:
            lines = f.readlines()
            # convert each line to a list of floats
            mask_xyxy_list = [list(map(float, line.strip().strip('[]').split(', '))) for line in lines]
            print(mask_xyxy_list)
            size = (512, 512)

            for idx, mask_xyxy in enumerate(mask_xyxy_list):
                # Create mask
                img = Image.new("RGB", size, (0, 0, 0))
                d = ImageDraw.Draw(img)
                d.rectangle(mask_xyxy, fill="white")
                img.save(args.mask_output_dir + '/m_' + name + '_' + str(idx) + '.jpg')

                # Create cutout
                if args.include_cutout:
                    img_path = glob.glob(args.source_dir + '/' + name + '.*')[0]
                    img = Image.open(img_path)
                    img = img.crop(mask_xyxy)
                    img.save(args.cutout_output_dir + '/c_' + name + '_' + str(idx) + '.jpg')


if __name__ == '__main__':
    main()
