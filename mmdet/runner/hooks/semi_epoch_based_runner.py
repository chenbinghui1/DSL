import os.path as osp
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines.formating import to_tensor
import platform
import shutil
import time
import warnings
import numpy as np
import cv2

import torch
from torch.optim import Optimizer
import logging
import copy
from collections import OrderedDict

import mmcv
from mmcv.runner import EpochBasedRunner,HOOKS
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.utils import get_host_info
from . import EMAOWNHook
import torch.distributed as dist



from mmcv.parallel import is_module_wrapper
from mmcv.runner.checkpoint import load_checkpoint
from mmcv.runner.dist_utils import get_dist_info
from mmcv.runner.log_buffer import LogBuffer
from mmcv.runner.priority import Priority, get_priority
from mmcv.runner.utils import get_time_str

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
P = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize_fix'),
    dict(type='PatchShuffle_fix'),
    dict(type='RandomFlip_fix'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),
]
img_scale=[(1333, 640), (1333, 800)]

@RUNNERS.register_module()
class SemiEpochBasedRunner(EpochBasedRunner):
    """SemiEpoch-based Runner.
    This runner train models epoch by epoch.
    """
    def __init__(self,
                 model,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None,
                 ema_model=None,
                 scale_invariant=False):
        if batch_processor is not None:
            if not callable(batch_processor):
                raise TypeError('batch_processor must be callable, '
                                f'but got {type(batch_processor)}')
            warnings.warn('batch_processor is deprecated, please implement '
                          'train_step() and val_step() in the model instead.')
            # raise an error is `batch_processor` is not None and
            # `model.train_step()` exists.
            if is_module_wrapper(model):
                _model = model.module
            else:
                _model = model
            if hasattr(_model, 'train_step') or hasattr(_model, 'val_step'):
                raise RuntimeError(
                    'batch_processor and model.train_step()/model.val_step() '
                    'cannot be both available.')
        else:
            assert hasattr(model, 'train_step')

        # check the type of `optimizer`
        if isinstance(optimizer, dict):
            for name, optim in optimizer.items():
                if not isinstance(optim, Optimizer):
                    raise TypeError(
                        f'optimizer must be a dict of torch.optim.Optimizers, '
                        f'but optimizer["{name}"] is a {type(optim)}')
        elif not isinstance(optimizer, Optimizer) and optimizer is not None:
            raise TypeError(
                f'optimizer must be a torch.optim.Optimizer object '
                f'or dict or None, but got {type(optimizer)}')

        # check the type of `logger`
        if not isinstance(logger, logging.Logger):
            raise TypeError(f'logger must be a logging.Logger object, '
                            f'but got {type(logger)}')

        # check the type of `meta`
        if meta is not None and not isinstance(meta, dict):
            raise TypeError(
                f'meta must be a dict or None, but got {type(meta)}')

        self.model = model
        self.batch_processor = batch_processor
        self.optimizer = optimizer
        self.logger = logger
        self.meta = meta
        # create work_dir
        if mmcv.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            mmcv.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()
        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

        if max_epochs is not None and max_iters is not None:
            raise ValueError(
                'Only one of `max_epochs` or `max_iters` can be set.')

        self._max_epochs = max_epochs
        self._max_iters = max_iters
        # TODO: Redesign LogBuffer, it is not flexible and elegant enough
        self.log_buffer = LogBuffer()
        self.ema_model = ema_model
        self.ema_flag = False
        self.imagefiles = []
        self.ITER=None
        self.iter_tol_epoch = 0
        self.scale_invariant = scale_invariant


    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            if self.ema_flag:
                outputs = self.ema_model.val_step(data_batch, self.optimizer, **kwargs)
            else:
                outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        if self.ema_flag:
            self.ema_model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        self.iter_tol_epoch = len(self.data_loader)
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.imagefiles=[]
            for fi in range(len(data_batch['img_metas'].data[0])):
                self.imagefiles.append(data_batch['img_metas'].data[0][fi]['filename'])

            # for add resized img for scale invariant learning; using the last image in batch to construct the scale invariant input
            if self.scale_invariant:
                #### using the strong aug image
                data_batch['img_metas'].data[0].append(copy.deepcopy(data_batch['img_metas'].data[0][-1]))
                data_batch['img_metas'].data[0][-1]['img_shape'] = (int(data_batch['img_metas'].data[0][-1]['img_shape'][0]/2), int(data_batch['img_metas'].data[0][-1]['img_shape'][1]/2), data_batch['img_metas'].data[0][-1]['img_shape'][2])
                data_batch['img_metas'].data[0][-1]['pad_shape'] = (int(data_batch['img_metas'].data[0][-1]['pad_shape'][0]/2), int(data_batch['img_metas'].data[0][-1]['pad_shape'][1]/2), data_batch['img_metas'].data[0][-1]['pad_shape'][2])
                data_batch['img_metas'].data[0][-1]['scale_factor'] = data_batch['img_metas'].data[0][-1]['scale_factor']/2

                tmp = torch.zeros_like(data_batch['img'].data[0][1:,:,:,:])
                _,_,h,w = data_batch['img'].data[0].shape
                cdata = torch.nn.functional.interpolate(data_batch['img'].data[0][1:,:,:,:].clone(), (int(h/2),int(w/2)), mode='bilinear')
                tmp[:,:,:int(h/2),:int(w/2)] = cdata
                data_batch['img'].data[0] = torch.cat((data_batch['img'].data[0],tmp),0)

                data_batch['gt_bboxes'].data[0].append(data_batch['gt_bboxes'].data[0][-1].clone()/2)
                data_batch['gt_labels'].data[0].append(data_batch['gt_labels'].data[0][-1].clone())
                data_batch['gt_bboxes_ignore'].data[0].append(data_batch['gt_bboxes_ignore'].data[0][-1].clone())
                if len(data_batch['gt_bboxes_ignore'].data[0][-1])>0:
                        data_batch['gt_bboxes_ignore'].data[0][-1]=data_batch['gt_bboxes_ignore'].data[0][-1]/2

                #### using the weak aug image
                #data_batch['img_metas'].data[0].append(copy.deepcopy(data_batch['img_metas'].data[0][-1]))
                #data_batch['img_metas'].data[0][-1]['img_shape'] = (int(data_batch['img_metas'].data[0][-1]['img_shape'][0]/2), int(data_batch['img_metas'].data[0][-1]['img_shape'][1]/2), data_batch['img_metas'].data[0][-1]['img_shape'][2])
                #data_batch['img_metas'].data[0][-1]['pad_shape'] = (int(data_batch['img_metas'].data[0][-1]['pad_shape'][0]/2), int(data_batch['img_metas'].data[0][-1]['pad_shape'][1]/2), data_batch['img_metas'].data[0][-1]['pad_shape'][2])
                #data_batch['img_metas'].data[0][-1]['scale_factor'] = data_batch['img_metas'].data[0][-1]['scale_factor']/2

                #img_bytes = mmcv.FileClient(**dict(backend='disk')).get(data_batch['img_metas'].data[0][-1]['filename'])
                #img = mmcv.imfrombytes(img_bytes, flag='color')
                ## resize
                #img, scale_factor = mmcv.imrescale(img, img_scale[data_batch['img_metas'].data[0][-1]['scale_idx']], return_scale=True, backend='cv2')
                ## patchsuffle
                #if data_batch['img_metas'].data[0][-1]['PS']==True:
                #    img=img.copy()
                #    h,w,c=img.shape
                #    place=data_batch['img_metas'].data[0][-1]['PS_place']
                #    mode=data_batch['img_metas'].data[0][-1]['PS_mode']
                #    if mode == 'flip':
                #            crop_h = h
                #            crop_w = min(int(round(w*place)), w)
                #            if crop_w == w or crop_w == 0:
                #                A=1
                #            else:
                #                img1 = mmcv.imcrop(img, np.array([0, 0, crop_w - 1, crop_h - 1]))
                #                img2 = mmcv.imcrop(img, np.array([crop_w, 0, w - 1, h - 1]))
                #                img = cv2.hconcat([img2, img1])
                #    elif mode == 'flop':
                #            crop_h = min(int(round(h*place)), h)
                #            crop_w = w
                #            if crop_h == h or crop_h == 0:
                #                A=2
                #            else:
                #                img1 = mmcv.imcrop(img, np.array([0, 0, w - 1, crop_h - 1]))
                #                img2 = mmcv.imcrop(img, np.array([0, crop_h, w - 1, h - 1]))
                #                img = cv2.vconcat([img2, img1])
                #    else:
                #            raise NotImplementedError
                ## flip
                #if data_batch['img_metas'].data[0][-1]['flip']:
                #    img = mmcv.imflip(img, direction=data_batch['img_metas'].data[0][-1]['flip_direction'])
                ## norm & pad
                #img = mmcv.imnormalize(img, data_batch['img_metas'].data[0][-1]['img_norm_cfg']['mean'], data_batch['img_metas'].data[0][-1]['img_norm_cfg']['std'], data_batch['img_metas'].data[0][-1]['img_norm_cfg']['to_rgb'])
                #img = mmcv.impad_to_multiple(img, 32, pad_val=0)
                #if len(img.shape) < 3:
                #    img = np.expand_dims(img, -1)
                #img = np.ascontiguousarray(img.transpose(2, 0, 1))
                #img = to_tensor(img)

                #tmp = torch.zeros_like(data_batch['img'].data[0][1:,:,:,:])
                #_,h,w = img.shape
                #cdata = torch.nn.functional.interpolate(img.unsqueeze(0), (int(h/2),int(w/2)), mode='bilinear')
                #tmp[:,:,:int(h/2),:int(w/2)] = cdata
                #data_batch['img'].data[0] = torch.cat((data_batch['img'].data[0],tmp),0)

                #data_batch['gt_bboxes'].data[0].append(data_batch['gt_bboxes'].data[0][-1].clone()/2)
                #data_batch['gt_labels'].data[0].append(data_batch['gt_labels'].data[0][-1].clone())
                #data_batch['gt_bboxes_ignore'].data[0].append(data_batch['gt_bboxes_ignore'].data[0][-1].clone())
                #if len(data_batch['gt_bboxes_ignore'].data[0][-1])>0:
                #        data_batch['gt_bboxes_ignore'].data[0][-1]=data_batch['gt_bboxes_ignore'].data[0][-1]/2
            # end
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        if self.ema_flag:
            self.ema_model.eval()
            print("using ema model eval")
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.
        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def load_checkpoint(self,
                        filename,
                        map_location='cpu',
                        strict=False,
                        revise_keys=[(r'^module.', '')]):

        self.logger.info('load checkpoint from %s', filename)
        if self.ema_model is not None:
            self.logger.info("[INFO] Load Done self.ema_model!")
            load_checkpoint(self.ema_model,filename,map_location,strict,self.logger,revise_keys=revise_keys)
        return load_checkpoint(
            self.model,
            filename,
            map_location,
            strict,
            self.logger,
            revise_keys=revise_keys)

    @torch.no_grad()
    def EMA(self, keep_rate=0.1, mode='epoch', start_point=5):
        # wait for checkpoint saving done!
        dist.barrier()
        filename = 'epoch_{}.pth' if mode == 'epoch' else 'iteration_{}.pth'
        _idx = self.epoch if mode == 'epoch' else self.iter
        if self.ema_flag is False:
            self.logger.info("[INFO] Init self.ema_model!")
            tmp_model = copy.deepcopy(self.model)
            student_model_dict = tmp_model.state_dict()
            new_teacher_dict = OrderedDict()
            for key, value in self.ema_model.state_dict().items():
                if key in student_model_dict.keys():
                    new_teacher_dict[key] = (
                        student_model_dict[key].float() * (1 - keep_rate) + value.float() * keep_rate
                    )
                else:
                    raise Exception("{} is not found in student model".format(key))
            self.ema_model.load_state_dict(new_teacher_dict)
            self.ema_model.float()
            self.logger.info("[INFO] Init self.ema_model done!")
            self.ema_flag = True
            return
        #self.logger.info("[INFO] Update self.ema_model, keep_rate is %f",keep_rate)
        tmp_model = copy.deepcopy(self.model)
        tmp_model = tmp_model.float()
        student_model_dict = tmp_model.state_dict()
        self.ema_model.float()

        new_teacher_dict = OrderedDict()
        for key, value in self.ema_model.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.ema_model.load_state_dict(new_teacher_dict)
        #self.model.load_state_dict(self.ema_model.state_dict())
        #self.logger.info("[INFO] Update done!")
        dist.barrier()

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.
        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        if self.ema_flag:
                save_checkpoint(self.ema_model, filepath+'_ema')
                self.logger.info("[INFO] Save Done self.ema_model!")
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

    def register_ema_hook(self, ema_config):
        if ema_config is None:
            return
        if isinstance(ema_config, dict):
            ema_config.setdefault('type', 'EMAOWNHook')
            hook = mmcv.build_from_cfg(ema_config, HOOKS)
        else:
            hook = ema_config
        self.register_hook(hook, priority=45)

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                ema_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                momentum_config=None,
                                timer_config=dict(type='IterTimerHook'),
                                custom_hooks_config=None):
        """Register default and custom hooks for training.
        Default and custom hooks include:
        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | LrUpdaterHook        | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | MomentumUpdaterHook  | HIGH (30)               |
        +----------------------+-------------------------+
        | OptimizerStepperHook | ABOVE_NORMAL (40)       |
        +----------------------+-------------------------+
        | CheckpointSaverHook  | NORMAL (50)             |
        +----------------------+-------------------------+
        | IterTimerHook        | LOW (70)                |
        +----------------------+-------------------------+
        | LoggerHook(s)        | VERY_LOW (90)           |
        +----------------------+-------------------------+
        | CustomHook(s)        | defaults to NORMAL (50) |
        +----------------------+-------------------------+
        If custom hooks have same priority with default hooks, custom hooks
        will be triggered after default hooks.
        """
        self.register_lr_hook(lr_config)
        self.register_momentum_hook(momentum_config)
        self.register_optimizer_hook(optimizer_config)
        if ema_config is not None:
            self.register_ema_hook(ema_config)
        self.register_checkpoint_hook(checkpoint_config)
        self.register_timer_hook(timer_config)
        self.register_logger_hooks(log_config)
        self.register_custom_hooks(custom_hooks_config)


