"""
SSUL
Copyright (c) 2021-present NAVER Corp.
MIT License
"""

import os
import sys
import torch.utils.data as data
import numpy as np
import json

import torch
from PIL import Image

from utils.tasks import get_dataset_list, get_tasks

classes = [
    "void", "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ", "windowpane",
    "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant",
    "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror",
    "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub",
    "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter",
    "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs",
    "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop",
    "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine",
    "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship",
    "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool",
    "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
    "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan",
    "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator",
    "glass", "clock", "flag"
]


def ade_cmap():
    cmap = np.zeros((256, 3), dtype=np.uint8)
    colors = [
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255]
    ]

    for i in range(len(colors)):
        cmap[i] = colors[i]

    return cmap.astype(np.uint8)

class ADESegmentation(data.Dataset):
    cmap = ade_cmap()
    def __init__(self,
                 opts,
                 image_set='train',
                 transform=None, 
                 cil_step=0, 
                 mem_size=0):

        self.root=opts.data_root        
        self.task=opts.task
        self.overlap=opts.overlap
        self.unknown=opts.unknown
        
        self.image_set = image_set
        self.transform = transform
        
        ade_root = './datasets/data/ade'
        
        if self.image_set == 'train' or self.image_set == 'memory':
            split = 'training'
        else:
            split = 'validation'
            
        image_dir = os.path.join(self.root, 'images', split)
        mask_dir = os.path.join(self.root, 'annotations', split)
        
        assert os.path.exists(mask_dir), "annotations not found"
            
        self.target_cls = get_tasks('ade', self.task, cil_step)
        self.target_cls += [255] # including ignore index (255)
            
        if image_set=='test':
            file_names = open(os.path.join(ade_root, 'val.txt'), 'r')
            file_names = file_names.read().splitlines()
            
        elif image_set == 'memory':
            for s in range(cil_step):
                self.target_cls += get_tasks('ade', self.task, s)
            
            memory_json = os.path.join(ade_root, 'memory.json')

            with open(memory_json, "r") as json_file:
                memory_list = json.load(json_file)

            file_names = memory_list[f"step_{cil_step}"]["memory_list"]
            print("... memory list : ", len(file_names), self.target_cls)
            
            while len(file_names) < opts.batch_size:
                file_names = file_names * 2
                
        else:
            file_names = get_dataset_list('ade', self.task, cil_step, image_set, self.overlap)

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        self.file_names = file_names
        
        # class re-ordering
        all_steps = get_tasks('ade', self.task)
        all_classes = []
        for i in range(len(all_steps)):
            all_classes += all_steps[i]
            
        self.ordering_map = np.zeros(256, dtype=np.uint8) + 255
        self.ordering_map[:len(all_classes)] = [all_classes.index(x) for x in range(len(all_classes))]
        
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        file_name = self.file_names[index]
        
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        # sal_map is useless for ADE20K
        sal_map = Image.fromarray(np.ones(target.size[::-1], dtype=np.uint8))
        
        # re-define target label according to the CIL case
        target = self.gt_label_mapping(target)
        
        if self.transform is not None:
            img, target, sal_map = self.transform(img, target, sal_map)
        
        # add unknown label, background index: 0 -> 1, unknown index: 0
        if self.image_set == 'train' and self.unknown:
            
            target = torch.where(target == 255, 
                                 torch.zeros_like(target) + 255,  # keep 255 (uint8)
                                 target+1) # unknown label
            
            unknown_area = (target == 1)
            target = torch.where(unknown_area, torch.zeros_like(target), target)

        return img, target.long(), sal_map, file_name


    def __len__(self):
        return len(self.images)
    
    def gt_label_mapping(self, gt):
        gt = np.array(gt, dtype=np.uint8)
        if self.image_set != 'test':
            gt = np.where(np.isin(gt, self.target_cls), gt, 0)
        gt = self.ordering_map[gt]
        gt = Image.fromarray(gt)
        
        return gt
    
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

