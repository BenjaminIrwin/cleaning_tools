import os
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='data/images', help='path to images')
parser.add_argument('--mask_dir', type=str, default='out')
parser.add_argument('--cutout_dir', type=str, default='out')
parser.add_argument('--boundings_dir', type=str, default='out')
parser.add_argument('--caption_dir', type=str, default='out')
parser.add_argument('--prefix', type=str, default='img', help='prefix of images')


def main():
    args = parser.parse_args()
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    cutout_dir = args.cutout_dir
    boundings_dir = args.boundings_dir
    caption_dir = args.caption_dir

    # Get all the images
    image_files = glob.glob(image_dir + '/*')
    print('Found {} image files'.format(len(image_files)))

    # Get all the masks
    mask_files = glob.glob(mask_dir + '/*')
    print('Found {} mask files'.format(len(mask_files)))

    # Get all the cutouts
    cutout_files = glob.glob(cutout_dir + '/*')
    print('Found {} cutout files'.format(len(cutout_files)))

    # Get all the boundings
    boundings_files = glob.glob(boundings_dir + '/*')
    print('Found {} boundings files'.format(len(boundings_files)))

    # Get all the captions
    caption_files = glob.glob(caption_dir + '/*')
    print('Found {} caption files'.format(len(caption_files)))

    # Iterate through all the images
    for idx, image_file in enumerate(image_files):
        # Get the image name
        name = os.path.basename(image_file).replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
        print('Processing {}'.format(name))

        # Get the mask file
        mask_files = glob.glob(mask_dir + '/m_' + name + '_*')
        print('Found mask file {}'.format(mask_files))

        # Get the cutout file
        cutout_files = glob.glob(cutout_dir + '/c_' + name + '_*')
        print('Found cutout file {}'.format(cutout_files))

        # Get the boundings file
        boundings_file = glob.glob(boundings_dir + '/' + name)[0]
        print('Found boundings file {}'.format(boundings_file))

        # Get the caption file
        caption_files = glob.glob(caption_dir + '/t_' + name + '_*')
        print('Found caption file {}'.format(len(caption_files)))

        # Rename the files
        os.rename(image_file, image_file.replace(name, 'img_' + str(idx)))
        os.rename(boundings_file, boundings_file.replace(name, 'bnd_' + str(idx)))
        for mask_file in mask_files:
            os.rename(mask_file, mask_file.replace('m_' + name, 'msk_' + str(idx)))
        for cutout_file in cutout_files:
            os.rename(cutout_file, cutout_file.replace('c_' + name, 'cut_' + str(idx)))
        for caption_file in caption_files:
            os.rename(caption_file, caption_file.replace('t_' + name, 'cap_' + str(idx)))


if __name__ == '__main__':
    main()
