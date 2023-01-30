import argparse
import os

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--source_dir', type=str, default='images', help='Directory containing the source images')


def main():
    args = parser.parse_args()
    source_dir = args.source_dir

    to_delete = []

    # Iterate through all files in the source directory
    for file in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file)
        try:
            # Try to open the file
            Image.open(file_path).verify()
        except (IOError, SyntaxError) as e:
            print('Bad file:', file_path)
            to_delete.append(file_path)

            # Ask user for confirmation
    num_files = len(to_delete)
    print('Would you like to delete {} files?'.format(num_files))
    print('This cannot be undone.')
    answer = input('y/n: ')
    if answer == 'y':
        for file in to_delete:
            os.remove(file)
        print('Deleted {} files'.format(num_files))
    else:
        print('Aborted')


if __name__ == '__main__':
    main()
