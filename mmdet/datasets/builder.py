import copy
import platform
import random
import torch
from functools import partial
import torch.nn.functional as F

import numpy as np
from mmcv.parallel import collate, DataContainer
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader

from .samplers import DistributedGroupSampler, DistributedSampler, GroupSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def _concat_dataset(cfg, default_args=None):
    from .dataset_wrappers import ConcatDataset
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)
    separate_eval = cfg.get('separate_eval', True)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        # pop 'separate_eval' since it is not a valid key for common datasets.
        if 'separate_eval' in data_cfg:
            data_cfg.pop('separate_eval')
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets, separate_eval)


def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import (ConcatDataset, RepeatDataset,
                                   ClassBalancedDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = DistributedGroupSampler(
                dataset, samples_per_gpu, world_size, rank, seed=seed)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False, seed=seed)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_multi_dataloader(datasets,
                           imgs_per_gpu,
                           workers_per_gpu,
                           batch_config,
                           num_gpus=1,
                           dist=True,
                           seed=None,
                           **kwargs):
    return MultiDataLoader(datasets, imgs_per_gpu, workers_per_gpu, num_gpus, dist, batch_config, seed, **kwargs)


class MultiDataLoader(object):
    def __init__(self,
                 datasets,
                 imgs_per_gpu,
                 workers_per_gpu,
                 num_gpus,
                 dist,
                 batch_config,
                 seed,
                 **kwargs):
        self.datasets = datasets
        self.batch_cfg = batch_config
        self.imgs_per_gpu = imgs_per_gpu
        self.worker_per_gpu = workers_per_gpu
        self.num_gpus = num_gpus
        self.dist = dist
        self.concat = False
        self.multi_data_loaders = []
        self.iter_num = 0
        self.seed = seed
        # lee.lcy 2020.08.21
        self.golden_mix_finetuning = \
            batch_config.get('golden_mix_finetuning', False)
        if self.golden_mix_finetuning:
            self.epochSet = \
                batch_config.get('golden_config', dict()).get('epochSet', [1, 1])
            self.stepTimes = \
                batch_config.get('golden_config', dict()).get('stepTimes', 3)
            self.datasetsLen = [len(ds) for ds in self.datasets]
            assert len(self.epochSet) == len(self.datasetsLen) == 2, \
                "[ERROR] The length of epochSet and datasets must be equal to 2 for GoldenMixFinetuning."

            kwargs.update(dict(
                    golden_mix_finetuning=self.golden_mix_finetuning,
                    golden_config=batch_config.get('golden_config', dict()),
                ))

        self._parse_batch_config(**kwargs)

            # self.lrStep = \
            #     batch_config.get('golden_config', dict()).get('lrStep', 1)

    def __next__(self):
        if self.iter_num < self.__len__():
            self.iter_num += 1
            if self.concat:
                data = []
                for loader in self.multi_data_loaders:
                    data.append(loader.get_batch())
                data = self._merge_data2one_batch(data)
            else:
                data = []
                for loader in self.multi_data_loaders:
                    data.append(self._merge_data2one_batch(loader.get_batch()))
            return data
        else:
            self.iter_num = 0
            raise StopIteration

    def _parse_batch_config(self, **kwargs):
        if not self.golden_mix_finetuning:  # case of multi-dataloader for Sem
            ratio = self.batch_cfg['ratio']
        else:   # case of golden_mix_finetuning
            ratio = [[1.0*es*dl for es, dl in zip(self.epochSet, self.datasetsLen)] ]
        self.concat = True if len(ratio) == 1 else False
        for r in ratio:
            self.multi_data_loaders.append(
                _MultiDataLoader(self.datasets,
                                 r,
                                 self.imgs_per_gpu,
                                 self.worker_per_gpu,
                                 self.num_gpus,
                                 self.dist,
                                 self.seed,
                                 **kwargs)
            )

    def _merge_data2one_batch(self, data):
        if len(data) == 1 and isinstance(data[0], list):
           data = data[0]
        keys = data[0].keys()
        batch_data = {}
        for k in keys:
            batch_data[k] = []
        for info in data:
            for k in keys:
                for content in info[k].data[0]:
                    batch_data[k].append(content)

        stack_cpu_flag = {}
        for k in keys:
            stack_cpu_flag[k] = (data[0][k].stack, data[0][k].cpu_only)

        for k in keys:
            if stack_cpu_flag[k][0] and not stack_cpu_flag[k][1]:
                # bhchen add for different input sizes 07/30/2021
                if batch_data[k][0].shape[1] != batch_data[k][-1].shape[1] or batch_data[k][0].shape[2]!=batch_data[k][-1].shape[2]:
                        pad_1 = max(batch_data[k][0].shape[1],batch_data[k][-1].shape[1])
                        pad_2 = max(batch_data[k][0].shape[2],batch_data[k][-1].shape[2])
                        for i in range(int(len(batch_data[k]))):
                                batch_data[k][i] = F.pad(batch_data[k][i],(0,pad_2-batch_data[k][i].shape[2],0,pad_1-batch_data[k][i].shape[1]),'constant',0.0)
                        batch_data[k] = DataContainer([torch.stack(batch_data[k])], stack=True)
                else:
                        batch_data[k] = DataContainer([torch.stack(batch_data[k])], stack=True)
            elif not stack_cpu_flag[k][0] and not stack_cpu_flag[k][1]:
                batch_data[k] = DataContainer([batch_data[k]], stack=False)
            else:
                batch_data[k] = DataContainer([batch_data[k]], cpu_only=True)
        return batch_data

    def __iter__(self):
        return self

    def __len__(self):
        return max([len(dl) for dl in self.multi_data_loaders])


class _MultiDataLoader(object):
    def __init__(self,
                 datasets,
                 imgs_per_dataset_gpu,
                 batch_size,
                 workers_per_gpu,
                 num_gpus,
                 dist,
                 seed,
                 **kwargs):
        self.datasets = datasets
        self.total_batch_size = batch_size
        self.imgs_per_dataset_gpu = imgs_per_dataset_gpu
        self.workers_per_gpu = workers_per_gpu
        self.num_gpus = num_gpus
        self.dist = dist
        self.seed = seed
        # lee.lcy 2020.08.21
        self.golden_mix_finetuning = kwargs.pop('golden_mix_finetuning', False)
        self.golden_config = kwargs.pop('golden_config', dict())
        if self.golden_mix_finetuning:
            self.epochSet = self.golden_config.get('epochSet', [1, 1])
            self.stepTimes = self.golden_config.get('stepTimes', 3)
        # for k in ['golden_mix_finetuning', 'golden_config']:
        #     if k in kwargs: kwargs.pop(k)

        self.batch_sizes = self._parse_batch_sizes(imgs_per_dataset_gpu, batch_size)
        self._build_data_loaders(datasets, self.batch_sizes, **kwargs)

    def _build_data_loaders(self, datasets, batch_sizes, **kwargs):
        self.data_loaders = []
        self.data_iters = []
        for dataset, batch_size in zip(datasets, batch_sizes):
            batch_size = int(batch_size)
            if batch_size > 0:
                data_loader = build_dataloader(dataset,
                                               batch_size,
                                               self.workers_per_gpu,
                                               num_gpus=self.num_gpus,
                                               dist=self.dist,
                                               seed=self.seed,
                                               **kwargs)
                self.data_loaders.append(data_loader)
                self.data_iters.append(iter(data_loader))

    def _parse_batch_sizes(self, imgs_per_dataset_gpu, batch_size):
        imgs_per_dataset_gpu = np.array(imgs_per_dataset_gpu)
        # 四舍五入成整数
        if not self.golden_mix_finetuning:  # 普通的multi-dataloader，可能出现某一个dataset的出现数量为0
            per_dataset_batch = np.around((batch_size * imgs_per_dataset_gpu / np.sum(imgs_per_dataset_gpu)), decimals=0).astype(np.int)
        else:   # GoldenMixFinetuning，每一个dataset的出现数量至少为1
            per_dataset_batch = np.array([0, 0])
            bs1 = np.around(batch_size * imgs_per_dataset_gpu[0] / np.sum(imgs_per_dataset_gpu), decimals=0).astype(np.int)
            bs1 = np.min([np.max([1, bs1]),  batch_size-1])
            bs2 = batch_size - bs1
            per_dataset_batch[0], per_dataset_batch[1] = bs1, bs2
            per_dataset_batch = per_dataset_batch.astype(np.int)

        return per_dataset_batch

    def get_batch(self):
        batch_data = []
        for idx, data_iter in enumerate(self.data_iters):
            try:
                a = next(data_iter)
            except:
                self.data_iters[idx] = iter(self.data_loaders[idx])
                a = next(self.data_iters[idx])
            batch_data.append(a)
        return batch_data

    def __len__(self):
        if not self.golden_mix_finetuning:
            # bhchen 05/14/2021 change sum to max
            return max([len(dl) for dl in self.data_loaders])
        else:
            return np.ceil(np.max([len(dl)*self.epochSet[i]/self.stepTimes for i, dl in enumerate(self.data_loaders)])).astype(np.int)
