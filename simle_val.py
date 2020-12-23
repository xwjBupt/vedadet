from vedadet.datasets.pipelines.test_time_aug import MultiScaleFlipAug
import cv2
import argparse
import os
import os.path as osp
import numpy as np
import torch
import pdb

from vedacore.misc import Config, load_weights, ProgressBar, mkdir_or_exist

from vedadet.engines import build_engine
from vedacore.parallel import MMDataParallel


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default='/nas/xcode/vedadet/configs/trainval/tinaface/tinaface.py',
                        help='train config file path')
    parser.add_argument('--checkpoint', default='/nas/xcode/vedadet/tinaface_epoch_185_weights.pth',
                        help='checkpoint file')
    parser.add_argument('--outdir', default='/nas/xcode/vedadet/eval_dirs/tmp/test_tinaface/',
                        help='directory where widerface txt will be saved')

    args = parser.parse_args()
    return args


def toC(im_path, engine):
    engine.eval()
    results = {}
    img = cv2.imread(im_path)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)

    results['filename'] = im_path
    results['ori_filename'] = im_path
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['img_fields'] = ['img']

    img_scale = (1100, 1650)
    flip = False
    transforms = [
        dict(typename='Resize', keep_ratio=True),
        dict(typename='RandomFlip', flip_ratio=0.0),
        dict(typename='Normalize', **dict(
            mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)),
        dict(typename='Pad', size_divisor=32, pad_val=0),
        dict(typename='ImageToTensor', keys=['img']),
        dict(typename='Collect', keys=['img'])
    ]

    toc = MultiScaleFlipAug(transforms, img_scale, flip=flip)
    container = toc(results)
    container['img'][0] = container['img'][0].unsqueeze(0)
    temp = container['img_metas'][0]
    temp._data = [[temp._data]]
    container['img_metas'][0] = temp
    # container['img_metas'][0].data = [[container['img_metas'][0].data]]

    output = engine(container)
    return output[0][0], container['img'][0][0].cpu().detach().numpy().transpose(1,2,0)


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    mkdir_or_exist(osp.abspath(args.outdir))
    engine = build_engine(cfg.val_engine)
    load_weights(engine.model, args.checkpoint, map_location='cpu')
    device = torch.cuda.current_device()
    engine = MMDataParallel(
        engine.to(device), device_ids=[torch.cuda.current_device()])
    # engine, data_loader = prepare(cfg, args.checkpoint)

    dets, img = toC('/nas/xcode/vedadet/data/WIDERFace/WIDER_val/0--Parade/0_Parade_marchingband_1_1004.jpg', engine)

    print('dsfsdf')
