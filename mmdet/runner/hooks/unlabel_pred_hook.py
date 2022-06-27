import os
import time
import json
import math
import copy
import mmcv
import torch
import numpy as np

from mmcv.runner import Hook, HOOKS
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
#from mmdet.ops.nms import nms_wrapper
from mmcv.ops import nms
import torch.distributed as dist
from mmdet.datasets.api_wrappers import COCO
import json
import os,sys

def parse_det_results(results, score_thr, reverse_mapper=None):
    new_results = []
    for i, res_perclass in enumerate(results):
        class_id = i
        for per_class_results in res_perclass:
            xmin, ymin, xmax, ymax, score = per_class_results
            if score < score_thr:
                continue
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            dict_instance = dict()
            if reverse_mapper is not None and str(class_id) in set(reverse_mapper.keys()):
                dict_instance["category"] = reverse_mapper[str(class_id)]
            else:
                dict_instance["category"] = class_id
            dict_instance["category_index"] = class_id
            dict_instance["score"] = round(float(score), 6)
            dict_instance["bbox"] = [xmin, ymin, xmax, ymax]
            new_results.append(dict_instance)
    return new_results

def gen_save_json_dict(info_dict, save_root, reverse_mapper=None, save_polygon=False):
    task_type = info_dict["task_type"]
    dict_tmp = dict()
    for k in info_dict.keys():
        if k == "result":
            continue
        dict_tmp[k] = info_dict[k]
    # dict_tmp["infer_results"] = []
    score_thr = 0
    if task_type == 'Det':
        if "infer_score_thre" in set(info_dict.keys()):
            score_thr = max(info_dict["infer_score_thre"], score_thr)
            dict_tmp["infer_results"] = parse_det_results(info_dict["result"], score_thr, reverse_mapper)
        dict_tmp["infer_results"] = sorted(dict_tmp["infer_results"], key=lambda s: s["score"], reverse=True)
    else:
        raise Exception()

    return dict_tmp

def get_image_list_from_list(list_file, root, anno_root):
    coco = COCO(list_file)
    img_ids = coco.get_img_ids()
    image_path_list = []
    for i in img_ids:
        info = coco.load_imgs([i])[0]
        name = info['file_name'] + '.json'
        with open(os.path.join(anno_root, name),'r') as f:
            data = json.load(f)
            if min(info['width'], info['height']) >=32 and data['targetNum']>0:
                image_path_list.append(os.path.join(root, info['file_name']))
    num_images = len(image_path_list)
    if num_images == 0:
        print("[ERROR][ModelInfer] Found no image in {}".format(root))
        sys.exit()
    else:
        return image_path_list


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def save_results2file(result, image_path, image_height, image_width,
                      save_file_format, checkpoint_name, infer_score_thre,
                      id2cat, cat2id, image_root_path, save_root_path,
                      task_type, vis=False, save_polygon=False, anno_root_path=None, iou=0.1, fuse=False, first_ignore=False):
    # zhoujing.zjh 防止image_root_path以/结尾导致图像层级目录个数判断错误，导致保存的结果中缺少dataset
    image_root_path = os.path.abspath(image_root_path)
    image_name = image_path.split('/')[-1]
    sub_path = image_path.replace(image_root_path, "")
    sub_dirs = sub_path.split("/")[:-1]
    #assert 3 > len(sub_dirs) > 0, \
    #    "[Error][ModelInfer] Please check image_root_path for inference."
    for sub_dir in sub_dirs:
        #image_name = os.path.join(sub_dir, image_name)
        save_path = os.path.join(save_root_path, sub_dir)
        create_dir(save_path)

    if save_file_format == "json":
        save_path = os.path.join(save_path, image_name + ".json")
        info_dict = dict()
        if len(sub_dirs) == 2:
            info_dict["dataset_name"] = sub_dirs[-1]
        else:
            info_dict["dataset_name"] = ''
        info_dict["checkpoint_name"] = checkpoint_name
        info_dict["image_name"] = image_name
        info_dict["image_height"] = image_height
        info_dict["image_width"] = image_width
        info_dict["result"] = result
        info_dict["task_type"] = task_type
        info_dict["infer_score_thre"] = infer_score_thre
        dict_save = gen_save_json_dict(info_dict, save_root_path,
                                       reverse_mapper=id2cat,
                                       save_polygon=save_polygon)
        # bhchen fuse with old bboxes
        new_info=dict()
        new_info['targetNum'] = len(dict_save['infer_results'])
        new_info['rects'] = []
        new_info['scores'] = []
        new_info['tags'] = []
        new_info['cid'] = []
        for i in range(int(new_info['targetNum'])):
             if dict_save['infer_results'][i]['category'] not in cat2id.keys():
                     continue
             new_info['rects'].append(dict_save['infer_results'][i]['bbox'])
             new_info['scores'].append(dict_save['infer_results'][i]['score'])
             new_info['tags'].append(dict_save['infer_results'][i]['category'])
             new_info['cid'].append(dict_save['infer_results'][i]['category_index'])
        # load the old bboxes info
        with open(os.path.join(anno_root_path, sub_path[1:])+".json", 'r') as f:
             old_info = json.load(f)
        old_info['cid'] = []
        for i in range(int(old_info['targetNum'])):
             old_info['cid'].append(cat2id[old_info['tags'][i]])
        # to numpy array
        if fuse:
                if first_ignore:
                        bboxes = np.array(new_info['rects'], dtype=np.float32)
                        scores = np.array(new_info['scores'], dtype=np.float32)
                        cids = np.array(new_info['cid'], dtype=np.float32)
                else:
                        bboxes = np.array(old_info['rects'] + new_info['rects'], dtype=np.float32)
                        scores = np.array(old_info['scores'] + new_info['scores'], dtype=np.float32)
                        cids = np.array(old_info['cid'] + new_info['cid'], dtype=np.float32)
        else:
                bboxes = np.array(new_info['rects'], dtype=np.float32)
                scores = np.array(new_info['scores'], dtype=np.float32)
                cids = np.array(new_info['cid'], dtype=np.float32)
        #nms_op = getattr(nms_wrapper, 'nms')
        # fuse starting
        final_bboxes = []
        final_scores = []
        final_cids = []
        final_mask = []
        for i in range(0,len(id2cat)-1):
              tmp_scores = scores[cids==i]
              if len(tmp_scores) == 0:
                  continue
              tmp_bboxes = bboxes[cids==i,:]
              #cls_dets = np.concatenate((tmp_bboxes,tmp_scores[:,None]),axis=1)
              #cls_dets, _ = nms_op(cls_dets, iou_thr=iou)
              cls_dets, _ = nms(tmp_bboxes, tmp_scores, iou_threshold=iou, score_threshold=0.1)
              final_cids.extend([i]*cls_dets.shape[0])
              final_bboxes.extend(cls_dets[:,0:4].tolist())
              final_scores.extend(cls_dets[:,-1].tolist())
        final_info=dict()
        final_info["imageName"]=old_info["imageName"]
        final_info["targetNum"]=len(final_scores)
        final_info["rects"]=final_bboxes
        final_info["tags"]=[id2cat[str(i)] for i in final_cids]
        final_info["masks"]=[[] for j in range(len(final_scores))]
        final_info["scores"]=final_scores
        with open(save_path, "w", encoding='utf-8') as fopen:
            json.dump(final_info, fopen, indent=4, ensure_ascii=False)



class LoadImage(object):
    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # liangting.zl 08.20 add default pad_shape
        results['pad_shape'] = img.shape
        results['ori_filename'] = results['filename']
        return results

def inference_model(model, img, config, task_type, iou):
    """Inference image(s) with the model.

        Args:
            model (nn.Module): The loaded detector/segmentor.
            imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
                images.

        Returns:
            If imgs is a str, a generator will be returned, otherwise return the
            detection results directly.
        """
    # build the data pipeline
    test_pipeline = [LoadImage()] + config.data.unlabel_pred.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    ## bhchen add mirror testing 06/02/2021
    flip = config.data.unlabel_pred.get("eval_flip", False)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    image_height, image_width, _ = data['img_metas'][0].data['ori_shape']

    data = scatter(
        collate([data], samples_per_gpu=1),
        [torch.cuda.current_device()])[0]
    # forward the model
    with torch.no_grad():
        if task_type in {'Det', 'Sem'}:
            if flip:
                data_mirror = torch.flip(data['img'][0], [3])
                data_mirror_flop = torch.flip(data_mirror, [2])
                data_flop = torch.flip(data_mirror_flop, [3])
                data['img_metas'][0].append(data['img_metas'][0][0])
                data['img_metas'][0].append(data['img_metas'][0][0])
                data['img_metas'][0].append(data['img_metas'][0][0])
                data['img'][0] = torch.cat([data['img'][0], data_mirror, data_mirror_flop, data_flop], dim=0)
                result_tmp = model(return_loss = False, rescale = True, **data)
                result = result_tmp[0]
                result_mirror = result_tmp[1]
                result_mirror_flop = result_tmp[2]
                result_flop = result_tmp[3]
            else:
                result = model(return_loss=False, rescale=True, **data)[0]
        elif task_type == 'Cls':
            result = model(return_loss=False, **data)
        else:
            raise Exception()

    return result, image_height, image_width


def single_gpu_test(model, task_type, image_list,
                    image_root_path, config, id2cat, cat2id,
                    checkpoint_name, infer_score_thre,
                    save_root_path, save_file_format, anno_root_path=None,
                    vis=False, save_polygon=False, ema_model=None, iou=0.1, fuse=False, first_ignore=False):
    if ema_model is not None:
         model = ema_model
    model.eval()
    prog_bar = mmcv.ProgressBar(len(image_list))
    for idx in range(len(image_list)):
        image_path = image_list[idx]
        result, image_height, image_width = inference_model(model, image_path, config, task_type, iou=iou)
        save_results2file(result, image_path, image_height, image_width,
                          save_file_format, checkpoint_name, infer_score_thre,
                          id2cat, cat2id, image_root_path, save_root_path,
                          task_type, vis, save_polygon, anno_root_path=anno_root_path, iou=iou, fuse=fuse, first_ignore=first_ignore)

        batch_size = 1
        for _ in range(batch_size):
            prog_bar.update()


def multi_gpu_test(model, task_type, image_list,
                   image_root_path, config, id2cat,cat2id,
                   checkpoint_name, infer_score_thre,
                   save_root_path, save_file_format,
                   rank, world_size, anno_root_path=None, vis=False, save_polygon=False, ema_model=None, iou=0.1, fuse=False, first_ignore=False):
    if ema_model is not None:
         model = ema_model
             
    model.eval()
    lens = len(image_list)
    if rank == 0:
        if lens >8:
            prog_bar = mmcv.ProgressBar(len(image_list))
    for idx in range(rank, len(image_list), world_size):
        image_path = image_list[idx]
        result, image_height, image_width = inference_model(model, image_path, config, task_type, iou=iou)
        save_results2file(result, image_path, image_height, image_width,
                          save_file_format, checkpoint_name, infer_score_thre,
                          id2cat, cat2id, image_root_path, save_root_path,
                          task_type, vis, save_polygon, anno_root_path=anno_root_path, iou=iou, fuse=fuse, first_ignore=first_ignore)

        batch_size = world_size
        if rank == 0:
            if lens >8:
                for _ in range(batch_size):
                    prog_bar.update()
    dist.barrier()

def adathres(rank, flag, filename, id2cat, cat2id, input_list, input_path, settings):
    if rank !=0 or flag == False:
        return
    else:
        ranges = settings.get("ranges",[0.3,0.35])
        gamma1 = settings.get("gamma1",0.05)
        gamma2 = settings.get("gamma2",0.6)
        base = settings.get("base",0.3)
        cnt = 0
        cnt_ = 0
        dis = {}
        cum = {}
        for files in input_list:
                file = files.split('/')[-1]
                with open(os.path.join(input_path,file.strip()+'.json'),'r') as f:
                    data = json.load(f)
                    if data['targetNum']==0:
                            cnt+=1
                    for j,i in enumerate(data['tags']):
                            if i in cat2id.keys():
                                if not os.path.exists(filename):
                                    if data['scores'][j] >=0.3:
                                        cnt_ +=1
                                        if id2cat[str(cat2id[i])] not in dis.keys():
                                                dis[id2cat[str(cat2id[i])]]=1
                                                cum[id2cat[str(cat2id[i])]]=data['scores'][j]
                                        else:
                                                dis[id2cat[str(cat2id[i])]]+=1
                                                cum[id2cat[str(cat2id[i])]]+=data['scores'][j]
                                else:
                                    with open(filename,'r') as hist:
                                        history = json.load(hist)["thres"]
                                    if i not in history.keys():
                                        cnt_ +=1
                                        if id2cat[str(cat2id[i])] not in dis.keys():
                                                dis[id2cat[str(cat2id[i])]]=1
                                                cum[id2cat[str(cat2id[i])]]=data['scores'][j]
                                        else:
                                                dis[id2cat[str(cat2id[i])]]+=1
                                                cum[id2cat[str(cat2id[i])]]+=data['scores'][j]
                                        continue
                                    if data['scores'][j] >= history[i]:
                                        cnt_ +=1
                                        if id2cat[str(cat2id[i])] not in dis.keys():
                                                dis[id2cat[str(cat2id[i])]]=1
                                                cum[id2cat[str(cat2id[i])]]=data['scores'][j]
                                        else:
                                                dis[id2cat[str(cat2id[i])]]+=1
                                                cum[id2cat[str(cat2id[i])]]+=data['scores'][j]
        #  For class weights
        avg = 0
        per = {}
        for i in dis.keys():
            avg +=dis[i]
        for i in dis.keys():
            per[i]=(avg/len(dis)/cum[i])**gamma2
        final = {}
        for i in sorted(per):
            final[i]=per[i]
        id_final={}
        for i in final.keys():
            id_final[int(cat2id[i])]=final[i]
        # For class threshold
        per = {}
        for i in dis.keys():
            per[i]=max(min((cum[i]/(avg/len(dis)))**gamma1*base, ranges[1]), ranges[0])
        # output
        Final = dict()
        Final['cat'] = final
        Final['id'] = id_final
        Final['thres'] = per
        with open(filename,'w') as f:
            json.dump(Final, f,indent=4,ensure_ascii=False)


@HOOKS.register_module()
class UnlabelPredHook(Hook):

    def __init__(self, kwargs, config, task_type, interval_mode='epoch', interval=1):
        self.dataset_type = kwargs["type"]
        self.num_gpus = kwargs["num_gpus"]
        self.image_root_path = os.path.abspath(kwargs["image_root_path"])
        self.image_list_file = os.path.abspath(kwargs["image_list_file"])
        self.anno_root_path = os.path.abspath(kwargs["anno_root_path"])
        
        self.start_point = int(kwargs.get("start_point", 0))
        self.fuse = kwargs.get("fuse_history", False)
        self.iter_fuse_flag = False
        self.first_ignore = True if not kwargs.get("first_fuse", True) else False
        self.config = config
        self.use_ema = kwargs["use_ema"]
        self.category_info_path = kwargs["category_info_path"]
        # zhoujing.zjh 2021.01.28 兼容config中dict输入
        if isinstance(self.category_info_path, str):
            category_info_file = open(kwargs["category_info_path"])
            category_info = json.load(category_info_file)
            category_info_file.close()
            self.id2cat = category_info["id2cat"]
            self.cat2id = category_info["cat2id"]
        elif isinstance(self.category_info_path, dict):
            if "id2cat" in self.category_info_path:
                self.id2cat = self.category_info_path["id2cat"]
                self.cat2ed = self.category_info_path["cat2id"]
            else:
                raise RuntimeError('[UnlabelPredHook] train_config \"category_info_path\" is a dict, but not found \"id2cat\".')
        else:
            raise RuntimeError('[UnlabelPredHook] train_config \"category_info_path\" is not str or dict')
        self.infer_score_thre = kwargs["infer_score_thre"] \
            if task_type in ('Det', 'Det_Sem') else None
        self.first_score_thre = kwargs.get("first_score_thre", None)
        if self.first_score_thre == None:
                self.first_score_thre = self.config.get("infer_score_thre",0.1)
        self.save_file_format = kwargs["save_file_format"]
        self.save_dir = config.work_dir
        self.image_list = get_image_list_from_list(self.image_list_file, self.image_root_path, self.anno_root_path)
        self.interval_mode = interval_mode
        self.interval = interval
        self.eval_config = kwargs["eval_config"] \
            if task_type in ('Det', 'Det_Sem') else None
        self.eval_img_path = os.path.abspath(kwargs.get("img_path"))
        self.eval_img_resize_size = kwargs.get("img_resize_size")
        self.eval_low_level_scale = kwargs.get("low_level_scale")
        self.task_type = task_type
        save_polygon = kwargs.get("save_polygon")
        self.prefile = ""
        self.preload_num = kwargs.get("preload",10)
        # bhchen add 08/13/2021 for adathres computation
        self.adathres_compute = config.data.unlabel_train.get("thres", None)
        if isinstance(self.adathres_compute, str):
            self.adathres_compute = True
            self.filename = config.data.unlabel_train.get("thres")
            self.settings = config.data.unlabel_pred.get("ada_thres_weight_settings", dict())
        else:
            self.adathres_compute = False
            self.filename = None
            self.settings = None
        ####################################
        if save_polygon is not None and save_polygon:
            self.save_polygon = True
        else:
            self.save_polygon = False

    def every_n_epochs_with_startpoint(self, runner, n, start):
        if runner.epoch + 1 < start:
            return False
        return (runner.epoch + 1 - start) % n == 0 if n > 0 else False

    def every_n_iters_with_startpoint(self, runner, n, start):
        if runner.iter + 1 < start:
            return False
        return (runner.iter + 1 - start) % n == 0 if n > 0 else False

    def after_train_epoch(self, runner):
        self.prefile = ""
        adathres(runner.rank, self.adathres_compute, self.filename, self.id2cat, self.cat2id, self.image_list, self.anno_root_path, self.settings)
        if self.interval_mode == "epoch" and runner.epoch+1>=self.start_point:
            if not self.every_n_epochs_with_startpoint(runner, self.interval, self.start_point):
                return
            self.after_train_epoch_iter(runner)

    def after_train_iter(self, runner):
        if self.interval_mode == "iteration" and runner.iter+1>=(self.start_point*runner.iter_tol_epoch)+1:
            if not self.every_n_iters_with_startpoint(runner, self.interval, self.start_point):
                return
            if self.iter_fuse_flag == True:
                self.after_train_iter_func(runner)
            else:
                # the first fuse will be the same as epoch manner
                self.after_train_epoch_iter(runner)
                self.iter_fuse_flag=True
                next(runner.ITER)
                # handling the preload mechanism in torch.dataloader
                # runner.ITER is an independent iterator for iter-style pseudo-label update
                for i in range(int(self.preload_num)):
                    next(runner.ITER)

    def after_train_epoch_iter(self, runner):
        iter = runner.iter if self.interval_mode == 'iteration' else runner.epoch
        runner.logger.info("[INFO]  Unlabel pred starting!")
        preds_save_dir = self.anno_root_path
        checkpoint_name = self.interval_mode + "_{}.pth".format(iter + 1)
        #preds_dir_name = checkpoint_name.replace(".pth", "_unlabel_preds")
        #preds_save_dir = os.path.join(self.save_dir, preds_dir_name)
        # bhchen 05/26/2021 add ema_model
        ema_model = None
        if runner.ema_flag and self.use_ema:
             ema_model = runner.ema_model
        # bhchen add 07/08/2021 for setting the first thre score when ignore the first label-fuse phase
        if self.first_score_thre != None:
                infer_score_thre = self.first_score_thre
                self.first_score_thre = None
        else:
                infer_score_thre = self.infer_score_thre

        if self.num_gpus > 1:
                multi_gpu_test(runner.model, self.task_type, self.image_list,
                           self.image_root_path, self.config, self.id2cat, self.cat2id, checkpoint_name, infer_score_thre,
                           preds_save_dir, self.save_file_format,
                           runner.rank, runner.world_size, anno_root_path=self.anno_root_path, save_polygon=self.save_polygon, ema_model = ema_model, iou=self.eval_config['iou'][0], fuse=self.fuse, first_ignore = self.first_ignore)
                # bhchen add for adathres computation;
                #adathres(runner.rank, self.adathres_compute, self.filename, self.id2cat, self.cat2id, self.image_list, self.anno_root_path, self.settings)
        else:
                single_gpu_test(runner.model, self.task_type, self.image_list,
                            self.image_root_path, self.config, self.id2cat, self.cat2id,
                            checkpoint_name, infer_score_thre,
                            preds_save_dir, self.save_file_format, anno_root_path = self.anno_root_path, save_polygon=self.save_polygon, ema_model = ema_model, iou=self.eval_config['iou'][0], fuse=self.fuse, first_ignore = self.first_ignore)
                # bhchen add for adathres computation
                #adathres(0, self.adathres_compute, self.filename, self.id2cat, self.cat2id, self.image_list, self.anno_root_path, self.settings)
        runner.logger.info("[INFO]  Unlabel pred Done!")
        runner.model.train()
        if runner.ema_flag and self.use_ema:
            runner.ema_model.train()
        # after the first fuse , ignore flag should be changed to Flase
        if self.fuse and self.first_ignore:
                self.first_ignore = False


    def after_train_iter_func(self, runner):
        iter = runner.iter if self.interval_mode == 'iteration' else runner.epoch
        #runner.logger.info("[INFO]  Unlabel pred starting!")
        preds_save_dir = self.anno_root_path
        checkpoint_name = self.interval_mode + "_{}.pth".format(iter + 1)
        assert len(runner.imagefiles) == 2
        next_iter = None
        try:
            # sampler_seed has run the first next() func
            next_iter = next(runner.ITER)
        except:
            next_iter = None
        if next_iter == None:
            return
        #if len(self.prefile)>0:
        #        assert self.prefile == runner.imagefiles[-1]
        #print(runner.imagefiles[-1], self.image_list[next_iter],iter)
        #print(self.image_list[next_iter],iter)

        # bhchen 05/26/2021 add ema_model
        ema_model = None
        if runner.ema_flag and self.use_ema:
             ema_model = runner.ema_model
        # bhchen add 07/08/2021 for setting the first thre score when ignore the first label-fuse phase
        if self.first_score_thre != None:
                infer_score_thre = self.first_score_thre
                self.first_score_thre = None
        else:
                infer_score_thre = self.infer_score_thre

        if self.num_gpus > 1:
                multi_gpu_test(runner.model, self.task_type, [self.image_list[next_iter]]*self.num_gpus,
                           self.image_root_path, self.config, self.id2cat, self.cat2id, checkpoint_name, infer_score_thre,
                           preds_save_dir, self.save_file_format,
                           runner.rank, runner.world_size, anno_root_path=self.anno_root_path, save_polygon=self.save_polygon, ema_model = ema_model, iou=self.eval_config['iou'][0], fuse=self.fuse, first_ignore = self.first_ignore)
                # bhchen add for adathres computation
        else:
                single_gpu_test(runner.model, self.task_type, [self.image_list[next_iter]]*self.num_gpus,
                            self.image_root_path, self.config, self.id2cat, self.cat2id,
                            checkpoint_name, infer_score_thre,
                            preds_save_dir, self.save_file_format, anno_root_path = self.anno_root_path, save_polygon=self.save_polygon, ema_model = ema_model, iou=self.eval_config['iou'][0], fuse=self.fuse, first_ignore = self.first_ignore)
                # bhchen add for adathres computation
        #runner.logger.info("[INFO]  Unlabel pred Done!")
        runner.model.train()
        if runner.ema_flag and self.use_ema:
            runner.ema_model.train()
        # after the first fuse , ignore flag should be changed to Flase
        if self.fuse and self.first_ignore:
                self.first_ignore = False
        self.prefile = self.image_list[next_iter]
        return
