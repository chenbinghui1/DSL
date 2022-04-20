from ..builder import DETECTORS
from .single_stage_multi_head import SingleStageMultiHeadDetector


@DETECTORS.register_module()
class FCOSMultiHead(SingleStageMultiHeadDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 head_num,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FCOSMultiHead, self).__init__(backbone, neck, head_num, bbox_head, bbox_head_1, train_cfg,
                                   test_cfg, pretrained, init_cfg)
