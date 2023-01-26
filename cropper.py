import glob
import os

import numpy as np
import argparse

from PIL import Image

parser = argparse.ArgumentParser(description='Cropper')
parser.add_argument('--input_image_folder', type=str, required=True,
                    help='The folder containing the images to be cropped.')
parser.add_argument('--width', type=int, required=True,
                    help='The width of the cropped images.')
parser.add_argument('--height', type=int, required=True,
                    help='The height of the cropped images.')
parser.add_argument('--copy', type=bool, required=False, default=False)


def random_crop(image, width, height):
    """Crops the given PIL.Image at a random location.
    Args:
        width: Desired output width.
        height: Desired output height.
    Returns:
        A function that takes an image and returns a randomly cropped image.
    """
    image_width, image_height = image.size
    offset_width = max(0, image_width - width)
    offset_height = max(0, image_height - height)
    offset = (np.random.randint(0, offset_width + 1),
              np.random.randint(0, offset_height + 1))
    return image.crop((offset[0], offset[1], offset[0] + width, offset[1] + height))


def main():
    args = parser.parse_args()

    input_image_folder = args.input_image_folder
    width = args.width
    height = args.height
    copy = args.copy

    a = 0
    test_files = []
    while len(test_files) == 0 and a < 10:
        test_files = glob.glob(input_image_folder + '/*')

    print('FILES FOUND: ', len(test_files))
    to_remove = []
    for idx, im in enumerate(test_files):
        image = Image.open(im)
        image_width, image_height = image.size
        if not (image_width == width and image_height == height):
            if image_width < width or image_height < height:
                print('REMOVING: ', im)
                # Remove image
                to_remove.append(im)
            else:
                print('CROPPING: ', idx, ' OF ', len(test_files))
                if image_width > width or image_height > height:
                    # Resize image making smaller dimension equal to desired dimension
                    if image_width > image_height:
                        new_height = height
                        new_width = int(image_width * (height / image_height))
                    else:
                        new_width = width
                        new_height = int(image_height * (width / image_width))

                    image = image.resize((new_width, new_height), Image.LANCZOS)

                # Crop image
                image = random_crop(image, width, height)


                # Save image
                if copy:
                    image.save(im.replace('.jpg', '_cropped.jpg'))
                else:
                    image.save(im)

    s = input('Do you want to remove the images (' + str(len(to_remove)) + ') that are too small? y/n')
    if s == 'y':
        for im in to_remove:
            os.remove(im)


if __name__ == "__main__":
    main()
