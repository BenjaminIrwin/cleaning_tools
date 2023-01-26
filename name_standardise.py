import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='data/images', help='path to images')
parser.add_argument('--prefix', type=str, default='img', help='prefix of images')


def group_rename(directory, prefix):
    for i, f in enumerate(os.listdir(directory)):
        # Get file format
        file_ext = f.split('.')[-1]
        # Rename file
        os.rename(os.path.join(directory, f), os.path.join(directory, prefix + str(i) + '.' + file_ext))


def main():
    args = parser.parse_args()
    group_rename(args.dir, args.prefix)


if __name__ == '__main__':
    main()
