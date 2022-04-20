import warnings
from mmcv.ops.nms import batched_nms
import sys
import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean


@DETECTORS.register_module()
class SingleStageMultiHeadDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head_num=2,
                 bbox_head=None,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageMultiHeadDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        bbox_head_1.update(train_cfg=train_cfg)
        bbox_head_1.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.head_num=head_num-1
        for i in range(self.head_num):
            name = 'neck_'+str(i+1)
            setattr(self, name, build_neck(neck))
            name = 'bbox_head_'+str(i+1)
            setattr(self, name, build_head(bbox_head_1))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.nms_cfg = test_cfg.get('nms')

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            y = self.neck(x)
            y_1 = self.neck_1(x)
            #y_2 = self.neck_2(x)
        return y, y_1#, y_2

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageMultiHeadDetector, self).forward_train(img, img_metas)
        x, x_1 = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)

        ex_losses = []
        reg_loss = 0
        for i in range(self.head_num):
            name = 'bbox_head_'+str(i+1)
            tmp_head = getattr(self, name)
            if i ==0:
                ex_losses.append(tmp_head.forward_train(x_1, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore))
            #if i ==1:
            #    ex_losses.append(tmp_head.forward_train(x_2, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore))
            #for (name1, param1), (name2, param2) in zip(dict(tmp_head.named_parameters(recurse=True)).items(), dict(self.bbox_head.named_parameters(recurse=True)).items()):
            #        if "weight" in name1:
            #                #reg_loss+=torch.abs(torch.sum((param1/torch.norm(param1)) * (param2/torch.norm(param2))))
            #                reg_loss+=torch.sum((param1-param2)*(param1-param2))

        for key in losses.keys():
            for i in range(self.head_num):
                losses[key] = losses[key] + ex_losses[i][key] * 1.0
        #losses["reg_loss"] = reg_loss/self.head_num * 0.001
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat,feat_1= self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)

        det_bbox, det_label = results_list[0]
        for i in range(self.head_num):
            name = 'bbox_head_'+str(i+1)
            tmp_head = getattr(self, name)
            if i ==0:
                results_list_ex = tmp_head.simple_test(feat_1, img_metas, rescale=rescale)
            #if i==1:
            #    results_list_ex = tmp_head.simple_test(feat_2, img_metas, rescale=rescale)
            det_bbox_ex, det_label_ex = results_list_ex[0]

            det_bbox = torch.cat((det_bbox, det_bbox_ex), 0)
            det_label = torch.cat((det_label, det_label_ex),0)
        if len(det_bbox)!=0:
            dets, keep = batched_nms(det_bbox[:,0:4], det_bbox[:,4].contiguous(), det_label, self.nms_cfg)
            dets = dets[:100]
            keep = keep[:100]
            results_list = [tuple([dets, det_label[keep]])]


        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        # TODO:move all onnx related code in bbox_head to onnx_export function
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas)

        return det_bboxes, det_labels
