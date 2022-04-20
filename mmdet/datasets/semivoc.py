from collections import OrderedDict
import json
import xml.etree.ElementTree as ET
import mmcv
import os
import numpy as np
from PIL import Image
from .pipelines import Compose

from mmcv.utils import print_log

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .xml_style import XMLDataset
from .custom import CustomDataset


@DATASETS.register_module()
class SemiVOCDataset(CustomDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 ann_path='',
                 labelmapper='',
                 thres=None,):
        self.ann_file = ann_file
        self.ann_path = ann_path
        self.labelmapper = json.load(open(labelmapper,'r'))
        self.thres=thres
        self.default_thres = [0.1, 0.3]
        self.thres_list_by_class = {}
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()
        self.pipeline = Compose(pipeline)
        self.proposals = None
        self.year=2007
        self.min_size = None

    def load_annotations(self, ann_file):
        """Load annotation from json ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        with open(ann_file,'r') as f:
            lines=f.readlines()
            for line in lines:
                filename = line.strip()
                img = Image.open(os.path.join(self.img_prefix, filename))
                width, height = img.size
                data_infos.append(
                                dict(id=filename.replace('.jpg',''), filename=filename, width=width, height=height))

        return data_infos
    def get_cat_ids(self, idx):
        img_id = self.data_infos[idx]['id']
        gt_labels = []
        img_info = self.data_infos[idx]
        name = img_info['filename']+'.json'
        with open(os.path.join(self.ann_path, name),'r') as f:
            data = json.load(f)
            for i in range(int(data['targetNum'])):
                x1,y1,x2,y2 = data['rects'][i]
                inter_w = max(0, min(x2, img_info['width']) - max(x1, 0))
                inter_h = max(0, min(y2, img_info['height']) - max(y1, 0))
                if inter_w * inter_h == 0:
                    continue
                if x2-x1<1 or y2-y1<1:
                    continue
                bbox = [x1,y1,x2,y2]
                if 'scores' in data.keys() and self.thres is not None:
                    if isinstance(self.thres, str):
                        if not os.path.exists(self.thres):
                            if data['scores'][i] < float(self.default_thres[1]) and data['scores'][i]>= float(self.default_thres[0]):
                                gt_bboxes_ignore.append(bbox)
                            else:
                                gt_bboxes.append(bbox)
                                gt_labels.append(int(self.labelmapper['cat2id'][data['tags'][i]]))
                                gt_masks_ann.append(None)
                        else:
                            with open(self.thres, 'r') as f:
                                self.thres_list_by_class = json.load(f)["thres"]
                                if data['tags'][i] not in self.thres_list_by_class.keys():
                                     if data['scores'][i] < float(self.default_thres[1]) and data['scores'][i]>= float(self.default_thres[0]):
                                         gt_bboxes_ignore.append(bbox)
                                     else:
                                         gt_bboxes.append(bbox)
                                         gt_labels.append(int(self.labelmapper['cat2id'][data['tags'][i]]))
                                         gt_masks_ann.append(None)
                                else:
                                     if data['scores'][i] < float(self.thres_list_by_class[data['tags'][i]]) and data['scores'][i]>= float(self.default_thres[0]):
                                         gt_bboxes_ignore.append(bbox)
                                     else:
                                         gt_bboxes.append(bbox)
                                         gt_labels.append(int(self.labelmapper['cat2id'][data['tags'][i]]))
                                         gt_masks_ann.append(None)
                    else:
                        if data['scores'][i] < float(self.thres[1]) and data['scores'][i]>= float(self.thres[0]):
                                gt_bboxes_ignore.append(bbox)
                        else:
                                gt_bboxes.append(bbox)
                                gt_labels.append(int(self.labelmapper['cat2id'][data['tags'][i]]))
                                gt_masks_ann.append(None)

                else:
                    gt_labels.append(int(self.labelmapper['cat2id'][data['tags'][i]]))
        return gt_labels

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            name = img_info['filename']+'.json'
            with open(os.path.join(self.ann_path, name),'r') as f:
                data = json.load(f)
                if min(img_info['width'], img_info['height']) < min_size or data['targetNum']==0:
                    continue
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        img_info = self.data_infos[idx]
        name = img_info['filename']+'.json'
        with open(os.path.join(self.ann_path, name),'r') as f:
            data = json.load(f)
            for i in range(int(data['targetNum'])):
                x1,y1,x2,y2 = data['rects'][i]
                inter_w = max(0, min(x2, img_info['width']) - max(x1, 0))
                inter_h = max(0, min(y2, img_info['height']) - max(y1, 0))
                if inter_w * inter_h == 0:
                    continue
                if x2-x1<1 or y2-y1<1:
                    continue
                bbox = [x1,y1,x2,y2]
                if 'scores' in data.keys() and self.thres is not None:
                    if isinstance(self.thres, str):
                        if not os.path.exists(self.thres):
                            if data['scores'][i] < float(self.default_thres[1]) and data['scores'][i]>= float(self.default_thres[0]):
                                gt_bboxes_ignore.append(bbox)
                            else:
                                gt_bboxes.append(bbox)
                                gt_labels.append(int(self.labelmapper['cat2id'][data['tags'][i]]))
                                gt_masks_ann.append(None)
                        else:
                            with open(self.thres, 'r') as f:
                                self.thres_list_by_class = json.load(f)["thres"]
                                if data['tags'][i] not in self.thres_list_by_class.keys():
                                     if data['scores'][i] < float(self.default_thres[1]) and data['scores'][i]>= float(self.default_thres[0]):
                                         gt_bboxes_ignore.append(bbox)
                                     else:
                                         gt_bboxes.append(bbox)
                                         gt_labels.append(int(self.labelmapper['cat2id'][data['tags'][i]]))
                                         gt_masks_ann.append(None)
                                else:
                                     if data['scores'][i] < float(self.thres_list_by_class[data['tags'][i]]) and data['scores'][i]>= float(self.default_thres[0]):
                                         gt_bboxes_ignore.append(bbox)
                                     else:
                                         gt_bboxes.append(bbox)
                                         gt_labels.append(int(self.labelmapper['cat2id'][data['tags'][i]]))
                                         gt_masks_ann.append(None)
                    else:
                        if data['scores'][i] < float(self.thres[1]) and data['scores'][i]>= float(self.thres[0]):
                                gt_bboxes_ignore.append(bbox)
                        else:
                                gt_bboxes.append(bbox)
                                gt_labels.append(int(self.labelmapper['cat2id'][data['tags'][i]]))
                                gt_masks_ann.append(None)

                else:
                    gt_bboxes.append(bbox)
                    gt_labels.append(int(self.labelmapper['cat2id'][data['tags'][i]]))
                    gt_masks_ann.append(None)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            )

        return ann
    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.CLASSES
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou_thr in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
