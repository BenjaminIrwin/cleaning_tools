
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import glob
import os
import re
import shutil

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, required=True, help='path to images')
parser.add_argument('--mask_path', type=str, required=True, help='path to masks')
parser.add_argument('--caption_path', type=str, required=True, help='path to captions')
parser.add_argument('--mask_with_no_class_file', type=str, required=True, help='path to the mask_with_no_class_file txt file')
parser.add_argument('--target_path', type=str, default='out', help='output dir')
parser.add_argument('--target_res', type=int, default=512, help='target resultions for the output images')
parser.add_argument('--target_classes', type=str, default='person', help='Class you want to process')

def get_mask_region(mask):
    h, w = mask.shape
    # print(f'image shape: {h} {w}')

    x1 = 0
    for i in range(w):
        if not (mask[:, i] == 0).all():
            break
        x1 += 1

    x2 = 0
    for i in reversed(range(w)):
        if not (mask[:, i] == 0).all():
            break
        x2 += 1
    x2 = w-x2

    y1 = 0
    for i in range(h):
        if not (mask[i] == 0).all():
            break
        y1 += 1

    y2 = 0
    for i in reversed(range(h)):
        if not (mask[i] == 0).all():
            break
        y2 += 1
    y2 = h-y2

    return (int(x1), int(y1), int(x2), int(y2))


def get_crop_region(mask, pad=0, target_res=512):
    h, w = mask.shape
    # print(f'in function target res: {target_res}, padding: {pad}')
    crop_left, crop_top, crop_right, crop_bottom = get_mask_region(mask)
    # print(f'position of box: ({crop_left} {crop_top} {crop_right} {crop_bottom})')

    crop_left = crop_left - pad
    crop_top = crop_top - pad
    crop_right = crop_right + pad
    crop_bottom = crop_bottom + pad

    while crop_right - crop_left < target_res:
        crop_right += 1
        crop_left -= 1
    
    while crop_bottom - crop_top < target_res:
        crop_bottom += 1
        crop_top -= 1

    while crop_right - crop_left > target_res:
        crop_right -= 1
        crop_left += 1
        if (crop_right - crop_left) - target_res == 1:
            crop_left += 1
    
    while crop_bottom - crop_top > target_res:
        crop_bottom -= 1
        crop_top += 1
        if (crop_bottom - crop_top) - target_res == 1:
            crop_top += 1

    if crop_left < 0:
        crop_right = target_res
        crop_left = 0

    if crop_top < 0:
        crop_bottom = target_res
        crop_top = 0

    if crop_right > w:
        crop_right = w
        crop_left = w - target_res

    if crop_bottom > h:
        crop_bottom = h
        crop_top = h - target_res

    return (int((crop_left)), int((crop_top)), int((crop_right)), int((crop_bottom)))

def resize_image(h , w, monochannel_mask, pil_image, percent_decrease, target_res):
        aspect_ratio = h / w

        if h < w:
            new_h = max(int(h / (1+percent_decrease)), target_res)
            new_w = int(new_h / aspect_ratio)
        else:
            new_w = max(int(w / (1+percent_decrease)), target_res)
            new_h = int(new_w * aspect_ratio)
        # print(f'new target shape: {new_h} {new_w}')

        monochannel_mask = monochannel_mask.resize((new_w, new_h), resample=LANCZOS)
        pil_image = pil_image.resize((new_w, new_h), resample=LANCZOS)

        return monochannel_mask, pil_image

def increase_box_size(x1, y1, x2, y2, monochannel_mask, percent_box_increase):
    mask_w, mask_h = monochannel_mask.size
    
    box_w = x2 - x1
    box_h = y2 - y1
    
    new_box_h = box_h * (1+percent_box_increase)
    new_box_w = box_w * (1+percent_box_increase)

    new_x1 = max(x1 - (new_box_w-box_w)/2, 0)
    new_x2 = min(x2 + (new_box_w-box_w)/2, mask_w)
    new_y1 = max(y1 - (new_box_h-box_h)/2, 0)
    new_y2 = min(y2 + (new_box_h-box_h)/2, mask_h)

    draw = ImageDraw.Draw(monochannel_mask)
    draw.rectangle([(new_x1, new_y1), (new_x2, new_y2)], fill=255)
    # monochannel_mask.show()

    return monochannel_mask, new_box_h, new_box_w

def full_res_transform(padding, target_res, pil_image, pil_mask):
    monochannel_mask = pil_mask.convert('L')
    x1, y1, x2, y2 = get_mask_region(np.array(monochannel_mask))

    percent_box_increase = np.random.noncentral_chisquare(6, 1, 1)/60
    # print(f'the precentage increase is: {percent_box_increase}')
    monochannel_mask, box_h, box_w = increase_box_size(x1, y1, x2, y2, monochannel_mask , percent_box_increase)

    if box_h + padding > target_res or box_w + padding > target_res:
        # print('box is too large, resizing the image')
        img_w, img_h = pil_image.size
        percent_decrease = max(((box_h + padding) - target_res) / target_res,
            ((box_w + padding) - target_res) / target_res)
        # print(f'% decrease = {percent_decrease}')

        monochannel_mask, pil_image = resize_image(img_h, img_w, monochannel_mask, pil_image, percent_decrease, target_res)

    crop_region = get_crop_region(np.array(monochannel_mask),
                                  padding, target_res)
    # print(f'crop_region = {crop_region}')

    monochannel_mask = monochannel_mask.crop(crop_region)
    cropped_image = pil_image.crop(crop_region)

    return cropped_image, monochannel_mask

def get_basename(i, target_classes='person', remove_sub_number=True):
    sub = re.sub('\.[^/.]+$', '', os.path.basename(i))
    if sub.startswith('m_') or sub.startswith('t_') or sub.startswith(target_classes + '_'):
        sub_ = sub[2:]
        if remove_sub_number:
            # remove all chars after the last underscore
            sub_ = sub_.rsplit('_', 1)[0]
        return sub_
    return sub

def get_matching_images(filename, image_files, previous_image):  
    image = ''
    if get_basename(previous_image) == filename.rsplit('_', 1)[0]:
        image = previous_image
        # print(f'{image} hasnt changed')
    else:
        for i in image_files:
            if get_basename(i) == filename.rsplit('_', 1)[0]:
                image = i
                break
    return image

def main():
    args = parser.parse_args()

    image_path = args.image_path
    mask_path = args.mask_path
    caption_path = args.caption_path
    mask_with_no_class_file = args.mask_with_no_class_file
    target_path = args.target_path
    target_res = args.target_res
    target_classes = args.target_classes

    image_save_path = target_path + '/images/'
    mask_save_path = target_path + '/masks/'
    caption_save_path = target_path + '/captions/'

    for save_path in [image_save_path, mask_save_path, caption_save_path]:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    mask_with_no_class_list = []
    with open(mask_with_no_class_file, 'r') as f:
        mask_with_no_class_list = f.read().splitlines()
    mask_with_no_class_list = [get_basename(m, remove_sub_number=False) for m in mask_with_no_class_list]

    print(f'{len(mask_with_no_class_list)} images have no class {target_classes}')

    image_files = []
    mask_files = []
    [image_files.extend(glob.glob(f'{image_path}' + '/*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]
    [mask_files.extend(glob.glob(f'{mask_path}' + '/*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]
    caption_files = glob.glob(caption_path + '/*')
    print(f'found {len(image_files)} images, {len(mask_files)} masks and {len(caption_files)} captions')

    no_class_num, no_caption_num, no_image_num, too_small_num, copied_num = 0, 0, 0, 0, 0
    previous_image = ''

    for mask in tqdm(mask_files):
        # print(f'\n working on {mask}')
        filename = get_basename(mask, remove_sub_number=False)

        if filename in mask_with_no_class_list:
            print(f'no class in {filename}, skipping')
            no_class_num += 1
            continue

        if not os.path.isfile(caption_path + 't_' + filename + '.txt'):
            print(f'no caption match for {filename}, skipping')
            no_caption_num += 1
            continue

        try:
            image = get_matching_images(filename, image_files, previous_image)
            pil_image = Image.open(image)
            pil_mask = Image.open(mask)
            previous_image = image
        except:
            print(f'no image match for {filename}, skipping')
            no_image_num += 1
            continue

        if min(pil_image.size) < target_res:
            print(f'{image} is too small, skipping.')
            too_small_num += 1
            continue

        min_padding_threshold = min(pil_mask.size) * 0.1
        padding = int(max(125, min_padding_threshold))

        pil_image, pil_mask = full_res_transform(padding, target_res, pil_image, pil_mask)

        pil_image.save(image_save_path + target_classes + '_' + filename + '.jpg')
        pil_mask.save(mask_save_path + target_classes + '_' + filename + '.jpg')
        shutil.copyfile(caption_path + 't_' + filename + '.txt', caption_save_path + target_classes + '_' + filename + '.txt')
        copied_num += 1
    
    print(f'{copied_num} successfuly copied and {no_class_num} masks with no class, {no_caption_num} missing captions, {no_image_num} missing images and {too_small_num} images that were too small')

    unique_images = set([get_basename(i, target_classes=target_classes) for i in glob.glob(image_save_path + '/*')])
    print(f'{len(unique_images)} copied')
    
if __name__ == '__main__':
    main()
