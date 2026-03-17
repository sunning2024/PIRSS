#!/usr/bin/env bash
set -e

CONFIG="configs/acdc/semantic-segmentation/maskformer2_R50_acdc_bs16_90k.yaml"
NGPUS=2   # 你当前命令行里显示的是 num_gpus=2，就按 2 来

python train_net.py \
  --config-file $CONFIG \
  --num-gpus $NGPUS \
  OUTPUT_DIR output/acdc_semantic/maskformer2_R50_acdc_bs16_90k
