# Check if all images in directory are 512x512:
import argparse
import os

from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--source_dir', type=str, default='images', help='Directory containing the source images')


def main():
    args = parser.parse_args()
    source_dir = args.source_dir

    print('Checking images in', source_dir)

    # Iterate through all files in the source directory
    for file in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file)
        try:
            # Try to open the file
            img = Image.open(file_path)
            if img.size != (512, 512):
                print('Dims violation:', file_path + ' ' + str(img.size))
        except (IOError, SyntaxError) as e:
            print('Bad file:', file_path)


if __name__ == '__main__':
    main()
