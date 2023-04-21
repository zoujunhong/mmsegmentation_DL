# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from mmseg.registry import DATASETS, TRANSFORMS
from .basesegdataset import BaseSegDataset
from mmcv.transforms import BaseTransform
import numpy as np
import mmcv
from typing import Optional
import mmengine
import cv2
from torchvision import transforms as pth_transforms
import torch
from mmseg.structures import SegDataSample
from mmcv.transforms import to_tensor
from mmengine.structures import PixelData

@TRANSFORMS.register_module()
class LoadVideoSequenceFromFile(BaseTransform):
    """Load a video sequence from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_path
    - seg_map_path

    Args:
    """

    def __init__(self, resize=(640, 480)) -> None:
        self.resize = resize
        self.img_transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        orig_names = results['img_path']
        seg_names = results['seg_map_path']

        # NxHxWx3 (BGR)
        images = [cv2.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), self.resize, interpolation=cv2.INTER_LINEAR) for f in orig_names]
        # NxHxW 
        seg_maps = [cv2.resize(cv2.imread(f), self.resize, interpolation=cv2.INTER_NEAREST)[:,:,0]  for f in seg_names]
        seg_maps = torch.cat([to_tensor(map[None,...].astype(np.int64)) for map in seg_maps], dim=0)
        seg_maps[seg_maps == 0] = 255
        seg_maps = seg_maps - 1
        seg_maps[seg_maps == 254] = 255

        results['img_path'] = None
        results['seg_map_path'] = None
        results['inputs'] = torch.cat([torch.from_numpy(i).unsqueeze(0) for i in images], dim=0)
        
        

        data_sample = SegDataSample()
        
        gt_sem_seg_data = dict(data=torch.cat(seg_maps, dim=0))
        data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)
        
        results['data_samples'] = seg_maps

        return results

    def __repr__(self):
        repr_str = ""
        return repr_str


@DATASETS.register_module()
class VSPWDataset(BaseSegDataset):
    """VSPW dataset.
    """
    METAINFO = dict(
        classes=('others', 'wall', 'ceiling', 'door', 'stair', 'ladder', 'escalator', 'Playground_slide', 'handrail_or_fence', 'window', 'rail', 'goal', 'pillar', 'pole', 'floor', 'ground', 'grass', 'sand', 'athletic_field', 'road', 'path', 'crosswalk', 'building', 'house', 'bridge', 'tower', 'windmill', 'well_or_well_lid', 'other_construction', 'sky', 'mountain', 'stone', 'wood', 'ice', 'snowfield', 'grandstand', 'sea', 'river', 'lake', 'waterfall', 'water', 'billboard_or_Bulletin_Board', 'sculpture', 'pipeline', 'flag', 'parasol_or_umbrella', 'cushion_or_carpet', 'tent', 'roadblock', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'wheeled_machine', 'ship_or_boat', 'raft', 'airplane', 'tyre', 'traffic_light', 'lamp', 'person', 'cat', 'dog', 'horse', 'cattle', 'other_animal',
                 'tree', 'flower', 'other_plant', 'toy', 'ball_net', 'backboard', 'skateboard', 'bat', 'ball', 'cupboard_or_showcase_or_storage_rack', 'box', 'traveling_case_or_trolley_case', 'basket', 'bag_or_package', 'trash_can', 'cage', 'plate', 'tub_or_bowl_or_pot', 'bottle_or_cup', 'barrel', 'fishbowl', 'bed', 'pillow', 'table_or_desk', 'chair_or_seat', 'bench', 'sofa', 'shelf', 'bathtub', 'gun', 'commode', 'roaster', 'other_machine', 'refrigerator', 'washing_machine', 'Microwave_oven', 'fan', 'curtain', 'textiles', 'clothes', 'painting_or_poster', 'mirror', 'flower_pot_or_vase', 'clock', 'book', 'tool', 'blackboard', 'tissue', 'screen_or_television', 'computer', 'printer', 'Mobile_phone', 'keyboard', 'other_electronic_product', 'fruit', 'food', 'instrument', 'train'),
        palette=[
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [191, 0, 0],
            [64, 128, 0],
            [191, 128, 0],
            [64, 0, 128],
            [191, 0, 128],
            [64, 128, 128],
            [191, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 191, 0],
            [128, 191, 0],
            [0, 64, 128],
            [128, 64, 128],
            [22, 22, 22],
            [23, 23, 23],
            [24, 24, 24],
            [25, 25, 25],
            [26, 26, 26],
            [27, 27, 27],
            [28, 28, 28],
            [29, 29, 29],
            [30, 30, 30],
            [31, 31, 31],
            [32, 32, 32],
            [33, 33, 33],
            [34, 34, 34],
            [35, 35, 35],
            [36, 36, 36],
            [37, 37, 37],
            [38, 38, 38],
            [39, 39, 39],
            [40, 40, 40],
            [41, 41, 41],
            [42, 42, 42],
            [43, 43, 43],
            [44, 44, 44],
            [45, 45, 45],
            [46, 46, 46],
            [47, 47, 47],
            [48, 48, 48],
            [49, 49, 49],
            [50, 50, 50],
            [51, 51, 51],
            [52, 52, 52],
            [53, 53, 53],
            [54, 54, 54],
            [55, 55, 55],
            [56, 56, 56],
            [57, 57, 57],
            [58, 58, 58],
            [59, 59, 59],
            [60, 60, 60],
            [61, 61, 61],
            [62, 62, 62],
            [63, 63, 63],
            [64, 64, 64],
            [65, 65, 65],
            [66, 66, 66],
            [67, 67, 67],
            [68, 68, 68],
            [69, 69, 69],
            [70, 70, 70],
            [71, 71, 71],
            [72, 72, 72],
            [73, 73, 73],
            [74, 74, 74],
            [75, 75, 75],
            [76, 76, 76],
            [77, 77, 77],
            [78, 78, 78],
            [79, 79, 79],
            [80, 80, 80],
            [81, 81, 81],
            [82, 82, 82],
            [83, 83, 83],
            [84, 84, 84],
            [85, 85, 85],
            [86, 86, 86],
            [87, 87, 87],
            [88, 88, 88],
            [89, 89, 89],
            [90, 90, 90],
            [91, 91, 91],
            [92, 92, 92],
            [93, 93, 93],
            [94, 94, 94],
            [95, 95, 95],
            [96, 96, 96],
            [97, 97, 97],
            [98, 98, 98],
            [99, 99, 99],
            [100, 100, 100],
            [101, 101, 101],
            [102, 102, 102],
            [103, 103, 103],
            [104, 104, 104],
            [105, 105, 105],
            [106, 106, 106],
            [107, 107, 107],
            [108, 108, 108],
            [109, 109, 109],
            [110, 110, 110],
            [111, 111, 111],
            [112, 112, 112],
            [113, 113, 113],
            [114, 114, 114],
            [115, 115, 115],
            [116, 116, 116],
            [117, 117, 117],
            [118, 118, 118],
            [119, 119, 119],
            [120, 120, 120],
            [121, 121, 121],
            [122, 122, 122],
            [123, 123, 123],
            [124, 124, 124]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 seq_length=5,
                 **kwargs) -> None:
        self.seq_length = seq_length
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def load_data_list(self):
        data_list = []
        lines = mmengine.list_from_file(
            self.ann_file, backend_args=self.backend_args)
        
        for line in lines:
            video_name = line.strip()
            mask_dir = os.path.join(self.data_root, "data", video_name, "mask")
            orig_dir = os.path.join(self.data_root, "data", video_name, "origin")
            if not os.path.isdir(mask_dir): continue # no label
            
            images = os.listdir(orig_dir)
            images.sort()
            for i in range(len(images)-self.seq_length+1):
                # base_name = img.split('.')[0]
                base_names = [images[j+i].split('.')[0] for j in range(self.seq_length)]
                data_info = {}
                data_info["img_path"] = [os.path.join(orig_dir, base_name+self.img_suffix) for base_name in base_names]
                data_info["seg_map_path"] = [os.path.join(mask_dir, base_name+self.seg_map_suffix) for base_name in base_names]

                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list