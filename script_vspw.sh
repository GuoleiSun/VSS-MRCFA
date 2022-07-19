#!/bin/bash
# 

# cd /cluster/home/celiuce/code2/SegFormer
cd /cluster/home/guosun/code/video-seg/SegFormer-code-mrcfa/SegFormer-shared

module load pigz
mkdir -p ${TMPDIR}/datasets_temp/
tar -I pigz -xvf /cluster/work/cvl/celiuce/video-seg/VSPW_480p.tar.gz -C ${TMPDIR}/datasets_temp/

# ls ${TMPDIR}/datasets/

model_path=/cluster/work/cvl/guosun/models/video-seg/segformer/
# model_path=/cluster/work/cvl/celiuce/video-seg/models/segformer/

# rsync -aq /cluster/home/celiuce/code2/SegFormer/ ${TMPDIR}
rsync -aq /cluster/home/guosun/code/video-seg/SegFormer-code-mrcfa/SegFormer-shared/ ${TMPDIR}
cd $TMPDIR

rm -r data/vspw/*
ln -s ${TMPDIR}/datasets_temp/VSPW_480p data/vspw/


# source /cluster/apps/local/env2lmod.sh && module load gcc/6.3.0 python_gpu/3.8.5
# source /cluster/project/cvl/admin/cvl_settings
# source /cluster/home/celiuce/det/bin/activate

source /cluster/home/guosun/envir/swav2/bin/activate


#### 

mkdir -p models/vspw2/

## save prediction results
--out models/vspw2/res.pkl --format-only --eval None

### apply interactive gpu

lmod2env && bsub -W 48:00 -R "rusage[mem=5000,ngpus_excl_p=4]" -R "select[gpu_mtotal0>=30000]"  -n 32 -Is bash -c "source /cluster/project/cvl/admin/cvl_settings; ${SHELL}"

# rm -r data/cityscapes/*
# ln -s ${TMPDIR}/datasets_temp/gtFine data/cityscapes/

### testing
# Single-gpu testing
# python tools/test.py local_configs/segformer/B1/segformer.b1.480x480.vspw.160k.py \
#     ${model_path}/vspw/work_dirs_4g_2//iter_24000.pth --out ${model_path}/vspw/work_dirs_4g_2/res.pkl

# # Multi-gpu testing
# ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw.160k.py \
#  ${model_path}/vspw/work_dirs_4g_2//iter_160000.pth 1 --out ${model_path}/vspw/work_dirs_4g_2/res.pkl

# # Multi-gpu, multi-scale testing
# tools/dist_test.sh local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /path/to/checkpoint_file <GPU_NUM> --aug-test


### training
# Single-gpu training
# python tools/train.py local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py --work-dir ${model_path}/work_dirs/

# Multi-gpu training, 4 gpus, b1
# ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw.160k.py 4 --work-dir ${model_path}/vspw/work_dirs_4g_2

# Multi-gpu training, 8 gpus, b4
# ./tools/dist_train.sh local_configs/segformer/B4/segformer.b4.480x480.vspw.160k.py 8 --work-dir ${model_path}/vspw/work_dirs_8g_b4

# Multi-gpu training, 4 gpus, b4
# ./tools/dist_train.sh local_configs/segformer/B4/segformer.b4.480x480.vspw.160k.py 4 --work-dir ${model_path}/vspw/work_dirs_4g_b4


## vspw2
# trainging for b0
./tools/dist_train.sh local_configs/segformer/B0/segformer.b0.480x480.vspw2_forFLOW_newPhotoMetricDistortion.160k.py 4 \
--work-dir ${model_path}/vspw2/work_dirs_4g_b0_bs2_num-clips4-963-depth1-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-5

./tools/dist_train.sh local_configs/segformer/B0/segformer.b0.480x480.vspw2_baseline.160k.py 4 \
--work-dir ${model_path}/vspw2/work_dirs_4g_b0_bs2_num-clips4-963-inverseVideoTrue-baseline            # b0 segformer baseline

./tools/dist_train.sh local_configs/segformer/B0/segformer.b0.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_b0_test_thre0.8_ensemble4_test

# trainging for b1
./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_baseline.160k.py 4 \
--work-dir ${model_path}/vspw2/work_dirs_4g_b1_bs2_num-clips4-963-inverseVideoTrue-baseline            # b1 segformer baseline

./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2.160k.py 4 \
 --work-dir ${model_path}/vspw2/work_test

./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_4

./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_ensemble4_clustering_3

./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_cluster.160k.py 4 \
 --work-dir ${model_path}/vspw2/cluster_num10_b1_depth4_swin_cluster_weight_3                    #  cluster_num10_b1_depth4_swin_token_selection_by_mask

./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_forFLOW_newPhotoMetricDistortion.160k.py 4 \
--work-dir ${model_path}/vspw2/work_dirs_4g_b1_bs2_num-clips4-963-depth2-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-7

# fast training
# ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_forFLOW_newPhotoMetricDistortion_fast.160k.py 4 \
# --work-dir ${model_path}/vspw2/work_dirs_flow_PhotoMetricDis_4g_b1_fast_bs2_num-clips4-963-depth2-oriWeight0.5-clips5_resize1_8_strip_pooling_liuyun_1by1_3dBN_flow_motion3-inverseVideoTrue

# trainging for b2
./tools/dist_train.sh local_configs/segformer/B2/segformer.b2.480x480.vspw2_baseline.160k.py 4 \
--work-dir ${model_path}/vspw2/work_dirs_4g_b2_bs2_num-clips4-963-inverseVideoTrue-baseline_2            # b2 segformer baseline

./tools/dist_train.sh local_configs/segformer/B2/segformer.b2.480x480.vspw2_hypercorr.160k.py 4 \
--work-dir ${model_path}/vspw2/hypercorr2_baseline_b2_affinityloss_w0.1_3

./tools/dist_train.sh local_configs/segformer/B2/segformer.b2.480x480.vspw2_hypercorr.160k.py 4 \
--work-dir ${model_path}/vspw2/topk_b2_test_thre0.7_ensemble4_2

./tools/dist_train.sh local_configs/segformer/B2/segformer.b2.480x480.vspw2_cluster.160k.py 4 \
 --work-dir ${model_path}/vspw2/cluster_num10_b2_depth4_swin_token_selection_by_mask_2

./tools/dist_train.sh local_configs/segformer/B2/segformer.b2.480x480.vspw2_forFLOW_newPhotoMetricDistortion.160k.py 4 \
--work-dir ${model_path}/vspw2/work_dirs_4g_b2_bs2_num-clips4-963-depth2-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-kernel755-seed1234-3

# trainging for b4
./tools/dist_train.sh local_configs/segformer/B4/segformer.b4.480x480.vspw2_baseline.160k.py 4 \
--work-dir ${model_path}/vspw2/work_dirs_4g_b4_bs2_num-clips4-963-inverseVideoTrue-baseline_2         # b4 segformer baseline

./tools/dist_train.sh local_configs/segformer/B4/segformer.b4.480x480.vspw2_forFLOW_newPhotoMetricDistortion.160k.py 4 \
--work-dir ${model_path}/vspw2/work_dirs_4g_b4_bs2_num-clips4-963-depth4-oriWeight0.5-clips_focal_trans3-inverseVideoTrue

./tools/dist_train.sh local_configs/segformer/B4/segformer.b4.480x480.vspw2_hypercorr.160k.py 4 \
--work-dir ${model_path}/vspw2/topk_b4_test_thre0.5_ensemble4

# trainging for b5
./tools/dist_train.sh local_configs/segformer/B5/segformer.b5.480x480.vspw2_baseline.160k.py 4 \
--work-dir ${model_path}/vspw2/work_dirs_4g_b5_bs2_num-clips4-963-inverseVideoTrue-baseline           # b5 segformer baseline

./tools/dist_train.sh local_configs/segformer/B5/segformer.b5.480x480.vspw2_forFLOW_newPhotoMetricDistortion.160k.py 4 \
--work-dir ${model_path}/vspw2/work_dirs_4g_b5_bs2_num-clips4-963-depth4-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-4

./tools/dist_train.sh local_configs/segformer/B5/segformer.b5.480x480.vspw2_forFLOW_newPhotoMetricDistortion_noflow.160k.py 4 \
--work-dir ${model_path}/vspw2/work_dirs_4g_b5_bs2_num-clips4-963-depth2-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue_noflow-kernel755-1

./tools/dist_train.sh local_configs/segformer/B5/segformer.b5.480x480.vspw2_hypercorr.160k.py 4 \
--work-dir ${model_path}/vspw2/topk_b5_test_thre0.5_ensemble4_2



# debug code
# ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_forFLOW_newPhotoMetricDistortion.160k.py 1 \
#  --work-dir ${model_path}/vspw2/work_dirs_flow_PhotoMetricDis_4g_b1_bs2_num-clips4-963-depth2-oriWeight0.5-clips5_resize1_8_strip_pooling_liuyun_1by1_3dBN_flow_motion3-inverseVideoTrue

# # debug code
# ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2.160k.py 1 \
#  --work-dir ${model_path}/vspw2/work_test

## vspw2 test 
./tools/dist_test.sh local_configs/segformer/B0/segformer.b0.480x480.vspw2_forFLOW_newPhotoMetricDistortion.160k.py \
  ${model_path}/vspw2/work_dirs_4g_b0_bs2_num-clips4-963-depth1-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-5/iter_160000.pth 4 \
   --out ${model_path}/vspw2/work_dirs_4g_b0_bs2_num-clips4-963-depth1-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-5/res.pkl

./tools/dist_test.sh local_configs/segformer/B0/segformer.b0.480x480.vspw2_baseline.160k.py \
  ${model_path}/vspw2/work_dirs_4g_b0_bs2_num-clips4-963-inverseVideoTrue-baseline/iter_160000.pth 4 \
   --out ${model_path}/vspw2/work_dirs_4g_b0_bs2_num-clips4-963-inverseVideoTrue-baseline/res.pkl

./tools/dist_test.sh local_configs/segformer/B0/segformer.b0.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_b0_test_thre0.8_ensemble4_3/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_b0_test_thre0.8_ensemble4_3/res.pkl

./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_3dconv/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_3dconv/res.pkl

./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_baseline.160k.py \
  ${model_path}/vspw2/work_dirs_4g_b1_bs2_num-clips4-963-inverseVideoTrue-baseline/iter_160000.pth 4 \
   --out ${model_path}/vspw2/work_dirs_4g_b1_bs2_num-clips4-963-inverseVideoTrue-baseline/res.pkl

./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_cluster.160k.py \
  ${model_path}/vspw2/cluster_num10_b1_depth4_swin_token_selection_by_mask_2/iter_160000.pth 4 \
   --out ${model_path}/vspw2/cluster_num10_b1_depth4_swin_token_selection_by_mask_2/res.pkl

./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2.160k.py \
  ${model_path}/vspw2/work_dirs_4g_b1_bs2_num-clips4-963-depth1-oriWeight0.5-clips2-inverseVideoTrue/iter_160000.pth 4 \
   --out ${model_path}/vspw2/work_dirs_4g_b1_bs2_num-clips4-963-depth1-oriWeight0.5-clips2-inverseVideoTrue/res.pkl

./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_forFLOW_newPhotoMetricDistortion.160k.py \
  ${model_path}/vspw2/work_dirs_4g_b1_bs2_num-clips4-963-depth2-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-7/iter_160000.pth 4 \
   --out ${model_path}/vspw2/work_dirs_4g_b1_bs2_num-clips4-963-depth2-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-7/res.pkl

./tools/dist_test.sh local_configs/segformer/B2/segformer.b2.480x480.vspw2_baseline.160k.py \
  ${model_path}/vspw2/work_dirs_4g_b2_bs2_num-clips4-963-inverseVideoTrue-baseline_2/iter_160000.pth 4 \
   --out ${model_path}/vspw2/work_dirs_4g_b2_bs2_num-clips4-963-inverseVideoTrue-baseline_2/res.pkl

./tools/dist_test.sh local_configs/segformer/B2/segformer.b2.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_b2_test_thre0.5_ensemble4_2/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_b2_test_thre0.5_ensemble4_2/res.pkl

./tools/dist_test.sh local_configs/segformer/B2/segformer.b2.480x480.vspw2_cluster.160k.py \
  ${model_path}/vspw2/cluster_num10_b2_depth4_swin_token_selection_by_mask/iter_156000.pth 4 \
   --out ${model_path}/vspw2/cluster_num10_b2_depth4_swin_token_selection_by_mask/res.pkl

./tools/dist_test.sh local_configs/segformer/B2/segformer.b2.480x480.vspw2_forFLOW_newPhotoMetricDistortion.160k.py \
  ${model_path}/vspw2/work_dirs_4g_b2_bs2_num-clips4-963-depth2-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-3/iter_160000.pth 4 \
   --out ${model_path}/vspw2/work_dirs_4g_b2_bs2_num-clips4-963-depth2-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-3/res.pkl

./tools/dist_test.sh local_configs/segformer/B4/segformer.b4.480x480.vspw2_baseline.160k.py \
  ${model_path}/vspw2/work_dirs_4g_b4_bs2_num-clips4-963-inverseVideoTrue-baseline_2/iter_160000.pth 4 \
   --out ${model_path}/vspw2/work_dirs_4g_b4_bs2_num-clips4-963-inverseVideoTrue-baseline_2/res.pkl

./tools/dist_test.sh local_configs/segformer/B4/segformer.b4.480x480.vspw2_forFLOW_newPhotoMetricDistortion.160k.py \
  ${model_path}/vspw2/work_dirs_4g_b4_bs2_num-clips4-963-depth4-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue/iter_160000.pth 4 \
   --out ${model_path}/vspw2/work_dirs_4g_b4_bs2_num-clips4-963-depth4-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue/res.pkl

./tools/dist_test.sh local_configs/segformer/B4/segformer.b4.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_b4_test_thre0.5_ensemble4/iter_156000.pth 4 \
   --out ${model_path}/vspw2/topk_b4_test_thre0.5_ensemble4/res.pkl

./tools/dist_test.sh local_configs/segformer/B5/segformer.b5.480x480.vspw2_baseline.160k.py \
  ${model_path}/vspw2/work_dirs_4g_b5_bs2_num-clips4-963-inverseVideoTrue-baseline_2/iter_160000.pth 4 \
   --out ${model_path}/vspw2/work_dirs_4g_b5_bs2_num-clips4-963-inverseVideoTrue-baseline_2/res.pkl

./tools/dist_test.sh local_configs/segformer/B5/segformer.b5.480x480.vspw2_forFLOW_newPhotoMetricDistortion.160k.py \
  ${model_path}/vspw2/work_dirs_4g_b5_bs2_num-clips4-963-depth4-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-3/iter_144000.pth 4 \
   --out ${model_path}/vspw2/work_dirs_4g_b5_bs2_num-clips4-963-depth4-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-3/res.pkl

./tools/dist_test.sh local_configs/segformer/B5/segformer.b5.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_b5_test_thre0.5_ensemble4/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_b5_test_thre0.5_ensemble4/res.pkl


## speed test for focal transformer

CUDA_VISIBLE_DEVICES=0 python speed.py local_configs/segformer/B0/segformer.b0.480x480.vspw2_forFLOW_newPhotoMetricDistortion_testtime.160k.py

CUDA_VISIBLE_DEVICES=0 python speed.py local_configs/segformer/B1/segformer.b1.480x480.vspw2_forFLOW_newPhotoMetricDistortion_testtime.160k.py

CUDA_VISIBLE_DEVICES=0 python speed.py local_configs/segformer/B2/segformer.b2.480x480.vspw2_forFLOW_newPhotoMetricDistortion_testtime.160k.py

CUDA_VISIBLE_DEVICES=0 python speed.py local_configs/segformer/B5/segformer.b5.480x480.vspw2_forFLOW_newPhotoMetricDistortion_testtime.160k.py

## speed test for hyper correlation
CUDA_VISIBLE_DEVICES=0 python speed.py local_configs/segformer/B0/segformer.b0.480x480.vspw2_hypercorr_testtime.160k.py

CUDA_VISIBLE_DEVICES=0 python speed.py local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr_testtime.160k.py

CUDA_VISIBLE_DEVICES=0 python speed.py local_configs/segformer/B2/segformer.b2.480x480.vspw2_hypercorr_testtime.160k.py

CUDA_VISIBLE_DEVICES=0 python speed.py local_configs/segformer/B4/segformer.b4.480x480.vspw2_hypercorr_testtime.160k.py

CUDA_VISIBLE_DEVICES=0 python speed.py local_configs/segformer/B5/segformer.b5.480x480.vspw2_hypercorr_testtime.160k.py


## gpu memory test for hyper correlation
CUDA_VISIBLE_DEVICES=0 python memory.py local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr_testtime.160k.py


## speed test for clustering

CUDA_VISIBLE_DEVICES=0 python speed.py local_configs/segformer/B1/segformer.b1.480x480.vspw2_cluster_testtime.160k.py


## vspw2 test 
# ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_forFLOW_newPhotoMetricDistortion.160k.py \
#   ${model_path}/vspw2/work_dirs_flow_PhotoMetricDis-lr_4g_b1_bs2_num-clips4-963-depth4-oriWeight0.5-clips2_resize_1_8-noFeatMerge-inverseVideoTrue-3/iter_160000.pth 4 \
#    --out ${model_path}/vspw2/work_dirs_flow_PhotoMetricDis-lr_4g_b1_bs2_num-clips4-963-depth4-oriWeight0.5-clips2_resize_1_8-noFeatMerge-inverseVideoTrue-3/res.pkl


# ## vspw2 test
# ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2.160k.py \
#  ${model_path}/vspw2/work_dirs_4g_b1_bs2_num-clips4-963-depth3-oriWeight0.5-clips5_resize1_8_strip_pooling_liuyun_1by1_3dBN-inverseVideoTrue/iter_160000.pth 4 \
#   --out ${model_path}/vspw2/work_dirs_4g_b1_bs2_num-clips4-963-depth3-oriWeight0.5-clips5_resize1_8_strip_pooling_liuyun_1by1_3dBN-inverseVideoTrue/res.pkl

 # python tools/test.py local_configs/segformer/B1/segformer.b1.480x480.vspw2.160k.py \
 #    ${model_path}/vspw2/work_dirs_1g_b1_noFeatMerge//iter_24000.pth --out ${model_path}/vspw2/work_dirs_1g_b1_noFeatMerge/res.pkl


##### Ablation study #####
./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr_ablation_reference321.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference321

 ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr_ablation_reference1.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference1

  ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr_ablation_reference3.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference3

   ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr_ablation_reference6.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference6

    ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr_ablation_reference9.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference9

     ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr_ablation_reference96.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference96_2

     ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr_ablation_reference63.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference63

./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre1.0_121_ensemble4

 ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre1.0_121_ensemble4_2

  ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre1.0_121_ensemble4_3

./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.9_121_ensemble4

./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.9_121_ensemble4_2

./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.7_121_ensemble4

 ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.7_121_ensemble4_2

  ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.7_121_ensemble4_3

 ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.7_121_ensemble4_4

 ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.3_121_ensemble4

 ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.1_121_ensemble4


  ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_onlysar

   ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_onlycfm

  ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_L1_2

   ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_L2_2

./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_top1

./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_top10

 ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_top50

  ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_top100

 ./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py 4 \
 --work-dir ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_topall



 ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr_ablation_reference321.160k.py \
  ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference321/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference321/res.pkl

    ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr_ablation_reference1.160k.py \
  ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference1/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference1/res.pkl

       ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr_ablation_reference3.160k.py \
  ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference3/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference3/res.pkl

          ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr_ablation_reference6.160k.py \
  ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference6/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference6/res.pkl

   ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr_ablation_reference9.160k.py \
  ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference9/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference9/res.pkl

   ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr_ablation_reference96.160k.py \
  ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference96/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference96/res.pkl

   ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr_ablation_reference63.160k.py \
  ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference63/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_reference63/res.pkl


   ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_test_thre1.0_121_ensemble4/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre1.0_121_ensemble4/res.pkl

   ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_test_thre0.9_121_ensemble4/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.9_121_ensemble4/res.pkl

   ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_test_thre0.7_121_ensemble4_5/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.7_121_ensemble4_5/res.pkl

   ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_test_thre0.3_121_ensemble4/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.3_121_ensemble4/res.pkl

   ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_test_thre0.1_121_ensemble4/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.1_121_ensemble4/res.pkl

   ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_onlysar/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_onlysar/res.pkl

   ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_onlycfm/iter_156000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_onlycfm/res.pkl

   ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_L1/iter_128000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_L1/res.pkl

   ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_L2_2/iter_128000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4L2_2/res.pkl

./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_top1/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_top1/res.pkl

   ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_top10/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_top10/res.pkl

   ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_top50/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_top50/res.pkl

   ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_top100/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_top100/res.pkl

   ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_topall/iter_160000.pth 4 \
   --out ${model_path}/vspw2/topk_test_thre0.5_121_ensemble4_topall/res.pkl