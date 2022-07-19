


./tools/dist_test.sh local_configs/mrcfa/B1/mrcfa.b1.480x480.vspw2_hypercorr.160k.py \
  ${model_path}/vspw2/hyper_baseline_FeaturePyramid2_baseline_b1/iter_160000.pth 4 \
   --out ${model_path}/vspw2/hyper_baseline_FeaturePyramid2_baseline_b1/res.pkl


./tools/dist_train.sh local_configs/mrcfa/B1/mrcfa.b1.480x480.vspw2_hypercorr.160k.py 4  --work-dir ${model_path}/vspw2/work_test