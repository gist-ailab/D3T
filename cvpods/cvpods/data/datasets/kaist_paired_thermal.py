#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import copy
import os
import os.path as osp
import xml.etree.ElementTree as ET
import megfile

import numpy as np

import torch

from cvpods.structures import BoxMode
from PIL import Image
from ..base_dataset import BaseDataset
from ..detection_utils import (
    annotations_to_instances,
    check_image_size,
    create_keypoint_hflip_indices,
    filter_empty_instances,
    read_image
)
from ..registry import DATASETS
from .paths_route import _PREDEFINED_SPLITS_KaistPairedThermal


"""
This file contains functions to parse ImageNet-format annotations into dicts in "cvpods format".
"""


@DATASETS.register()
class KaistPairedThermal(BaseDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(KaistPairedThermal, self).__init__(cfg, dataset_name, transforms, is_train)

        image_root, split = _PREDEFINED_SPLITS_KaistPairedThermal["voc"][self.name]
        self.image_root = osp.join(self.data_root, image_root) \
            if "://" not in image_root else image_root
        self.split = split

        self.meta = self._get_metadata()
        self.dataset_dicts = self._load_annotations()

        # fmt: off
        self.data_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.filter_empty = cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        self.proposal_files = cfg.DATASETS.PROPOSAL_FILES_TRAIN
        # fmt: on

        if is_train:
            self.dataset_dicts = self._filter_annotations(
                filter_empty=self.filter_empty,
                min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                if self.keypoint_on else 0,
                proposal_files=self.proposal_files if self.load_proposals else None,
            )
            self._set_group_flag()

        self.eval_with_gt = cfg.TEST.get("WITH_GT", False)

        if self.keypoint_on:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

    def __getitem__(self, index):
        """Load data, apply transforms, converto to Instances.
        """
        dataset_dict = copy.deepcopy(self.dataset_dicts[index])

        # read image
        image = read_image(dataset_dict["file_name"], format=self.data_format)
        check_image_size(dataset_dict, image)

        if "annotations" in dataset_dict:
            annotations = dataset_dict.pop("annotations")
            annotations = [
                ann for ann in annotations if ann.get("iscrowd", 0) == 0]
        else:
            annotations = None

        # apply transfrom
        image, annotations = self._apply_transforms(
            image, annotations)

        if annotations is not None:
            image_shape = image.shape[:2]  # h, w

            instances = annotations_to_instances(
                annotations, image_shape, mask_format=self.mask_format
            )

            # # Create a tight bounding box from masks, useful when image is cropped
            # if self.crop_gen and instances.has("gt_masks"):
            #     instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            dataset_dict["instances"] = filter_empty_instances(instances)

        # convert to Instance type
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        # h, w, c -> c, h, w
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))

        return dataset_dict

    def __len__(self):
        return len(self.dataset_dicts)

    def _get_metadata(self):
        # fmt: off
        thing_classes = ('person',)
        meta = {
            "thing_classes": thing_classes,
            "evaluator_type": _PREDEFINED_SPLITS_KaistPairedThermal["evaluator_type"]["voc"],
            "dirname": self.image_root,
            "split": self.split,
            "year": 2007,
            # "year": 2012,
        }
        return meta

    def get_path_from_id(self, img_id, image_type='lwir'):
        img_id = img_id.split('/')
        img_id = os.path.join(img_id[0], img_id[1], image_type, img_id[2])
        return img_id

    def get_img_info(self, image_path):
        img = Image.open(image_path).convert("RGB")
        im_info = tuple(map(int, (img.size[0], img.size[1]))) # Image: width, height
        return {"height": im_info[1], "width": im_info[0]}

    def _load_annotations(self):
        """
        Load Pascal VOC detection annotations to cvpods format.

        Args:
            dirname: Contain "Annotations", "ImageSets", "JPEGImages"
            split (str): one of "train", "test", "val", "trainval"
        """

        dirname = self.image_root
        split = self.split

        with megfile.smart_open(
            megfile.smart_path_join(dirname, "splits", split + ".txt")
        ) as f:
            fileids = np.loadtxt(f, dtype=np.str)

        dicts = []
        for fileid in fileids:
            anno_file = os.path.join(dirname, "annotations", self.get_path_from_id(fileid) + ".txt")
            jpeg_file = os.path.join(dirname, "images", self.get_path_from_id(fileid) + ".jpeg")
            if not os.path.exists(jpeg_file):
                jpeg_file = os.path.join(dirname, "images", self.get_path_from_id(fileid) + ".jpg")

            ######################
            size = self.get_img_info(jpeg_file)
            r = {
                "file_name": jpeg_file,
                "image_id": fileid,
                "height": size["height"],
                "width": size["width"],
            }

            instances = []

            # change cyclist ==> person
            with open(anno_file, 'r') as f:
                objs = f.readlines()[1:]
            remain_objs = []
            for obj in objs:
                obj_split = obj.split(' ')
                if obj_split[0].lower().strip() == 'person' and obj_split[5].strip() != '2':
                    remain_objs.append(obj)
                # cyclist as positive sample
                elif obj_split[0].lower().strip() == 'cyclist':
                    obj_split[0] = 'person'
                    remain_objs.append(' '.join(obj_split))
            objs = remain_objs

            for obj in objs:
                obj = obj.split(' ')
                x1 = float(obj[1]) - 1
                y1 = float(obj[2]) - 1
                x2 = x1 + float(obj[3])
                y2 = y1 + float(obj[4])
                bbox = [x1, y1, x2, y2]

                instances.append({
                    "category_id": CLASS_NAMES.index(obj[0].lower().strip()),
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS
                })
            #######################
            r["annotations"] = instances  #
            dicts.append(r)

        return dicts


# fmt: off
CLASS_NAMES = ('person',)
