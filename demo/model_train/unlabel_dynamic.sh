# for coco, copy the initial pseudo-labels to semicoco dir
anno_path="/gruntdata1/bhchen/factory/data/semicoco/unlabel_prepared_annos/Industry/annotations/full/"
rm -rf $anno_path
cp -r workdir_coco/r50_caffe_mslonger_tricks_0.1data/epoch_55.pth-unlabeled.bbox.json_thres0.1_annos/ $anno_path

# for voc, copy the initial pseudo-labels to semivoc dir
#rm -rf ../data/semivoc/unlabel_prepared_annos/Industry/annotations/full/
#cp -r workdir_voc/RLA_r50_caffe_mslonger_tricks_alldata/epoch_55.pth-unlabeled.bbox.json_thres0.1_annos/ ../data/semivoc/unlabel_prepared_annos/Industry/annotations/full/
echo "remove & copy annotations done!"

CONFIG=configs/fcos_semi/RLA_r50_caffe_mslonger_tricks_0.Xdata_unlabel_dynamic_lw_nofuse_iterlabel_si-soft_singlestage.py
WORKDIR=workdir_coco/0.1data
GPU=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29502 ./tools/dist_train.sh $CONFIG $GPU --work-dir $WORKDIR
