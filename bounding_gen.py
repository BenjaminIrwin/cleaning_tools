import glob
import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from numpy import random
from torchvision import transforms

sys.path.insert(0,'yolov7')

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, set_logging
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

parser = argparse.ArgumentParser()
parser.add_argument('--source_dir', type=str, default='/content/test', help='source')
parser.add_argument('--txt_output_dir', type=str, default='/content/output', help='txt_output_dir')


def PIL_to_tensor(pil_image):
    return transforms.ToTensor()(pil_image)


def detect(target_class=0, conf_thres=0.7, iou_thres=0.45, imgsz=640,
           source='/content/test', classes=None, weights='yolov7x.pt', agnostic_nms=True,
           trace=False, augment=True, cpu=False, save_dir='/content/output'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with torch.no_grad():
        # Initialize
        set_logging()
        if cpu:
            device = select_device('cpu')
        else:
            device = select_device('0')

        half = False

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            model = TracedModel(model, device, imgsz)

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()

        results = {}

        for path, img, im0s, vid_cap in dataset:
            p = Path(path)  # to Path
            txt_path = save_dir + '/' + p.stem
            if os.path.exists(txt_path):
                print('Path {} already exists, skipping'.format(txt_path))
                continue
            else:
                print('Processing {}'.format(txt_path))
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (
                    old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            img_results = []

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                txt_path = save_dir + '/' + p.stem
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    detected_classes = []
                    # Print results
                    for c in det[:, -1].unique():
                        detected_classes.append(int(c))

                    for *xyxy, conf, cls in reversed(det):
                        xyxy_float = [c.item() for c in xyxy]
                        if cls.item() == target_class:
                            img_results.append(xyxy_float)

            with open(txt_path, 'w') as f:
                for item in img_results:
                    line = '[' + ', '.join(str(x) for x in item) + ']'
                    f.write(line + '\n')


def main():
    args = parser.parse_args()

    detect(target_class=0, conf_thres=0.35, iou_thres=0.45, imgsz=512,
           source=args.source_dir, cpu=False, save_dir=args.txt_output_dir)


if __name__ == '__main__':
    main()
