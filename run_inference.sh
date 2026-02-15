export TOKENIZERS_PARALLELISM=false

GPU=0
CKPT=../z_ckpt/exp_dsr/checkpoints/step=02400.ckpt
START=1
END=0

python3 \
inference.py \
$GPU \
$CKPT $START $END \
