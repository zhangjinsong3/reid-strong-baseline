# encoding: utf-8
"""
@author:  zjs
@contact: jingsongzhang@sf-express.com
"""

import argparse
import os
import sys
from os import mkdir

import numpy as np
import torch
from torch.backends import cudnn

from config import cfg
from modeling import build_model
from utils.logger import setup_logger
from scipy.spatial.distance import cdist

from tqdm import tqdm


def make_test_data_loader(cfg):
    from data.transforms import build_transforms
    from data.datasets import init_dataset
    from data.datasets import ImageDataset
    from torch.utils.data import DataLoader
    from data.collate_batch import test_collate_fn

    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, no_id=True, verbose=False)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, no_id=True, verbose=False)

    # num_classes = dataset.num_gallery_pids
    num_classes = 1000

    gallery_set = ImageDataset(dataset.gallery, val_transforms, no_id=True)
    query_set = ImageDataset(dataset.query, val_transforms, no_id=True)
    gallery_loader = DataLoader(
        gallery_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=test_collate_fn
    )
    query_loader = DataLoader(
        query_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=test_collate_fn
    )
    return gallery_loader, query_loader, len(dataset.query), num_classes


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="./configs/softmax_triplet.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    # init dataloader
    gallery_loader, query_loader, num_query, num_classes = make_test_data_loader(
        cfg)
    print('num query: %d, num_classes: %d' % (num_query, num_classes))
    # build model and load checkpoint param
    model = build_model(cfg, num_classes)
    model.load_param(cfg.TEST.WEIGHT)

    model.eval().cuda()
    feats = []
    g_pids = []
    g_camids = []
    g_names = []
    q_names = []
    # ===== extract feats =====
    with torch.no_grad():
        print('extract query feats...')
        for batch in query_loader:
            data, _, _, paths = batch
            feat = model(data.cuda())
            feats.append(feat.cpu())
            q_names.extend(paths)
        # print('extract gallery feats...')
        # for batch in gallery_loader:
        #     data, pids, camids, paths = batch
        #     g_pids.extend(pids)
        #     g_camids.extend(camids)
        #     feat = model(data.cuda())
        #     feats.append(feat.cpu())
        #     g_names.extend(paths)

    # ===== init vars =====
    feats = torch.cat(feats, dim=0)  # cat feats because feats is batch-wised
    if cfg.TEST.FEAT_NORM == 'yes':
        feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # normalize feats
    print(feats)
    exit()
    qf = feats[:num_query]  # query feats
    gf = feats[num_query:]  # gallery feats
    # ===== calc euclidean distance between gallery feat and query feat =====
    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.cpu().numpy()
    # distmat = cdist(qf.numpy(), gf.numpy())

    distmat_sort = np.argsort(distmat, axis=1)

    import cv2
    for i, q_name in enumerate(q_names):
        print('%03d: %s' % (i, q_name), end=', ')
        save_img = cv2.resize(cv2.imread(os.path.join(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.NAMES.title(), 'query', q_name)), (128, 128))
        for j in range(10):
            print(g_names[distmat_sort[i, j]], end=', ')
            tmp = cv2.resize(cv2.imread(os.path.join(cfg.DATASETS.ROOT_DIR,
                                                     cfg.DATASETS.NAMES.title(),
                                                     'test',
                                                     g_names[distmat_sort[i, j]])), (128, 128))
            cv2.putText(tmp, '%.2f' % distmat[i, distmat_sort[i, j]], (2, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
            save_img = cv2.hconcat((save_img, tmp))
        print(' ')
        print(distmat[i, distmat_sort[i, :10]])
        cv2.imwrite(os.path.join(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.NAMES.title(), '%03d.jpg' % i), save_img)


if __name__ == '__main__':
    main()
