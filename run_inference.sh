export TOKENIZERS_PARALLELISM=false

GPU=0
CKPT=../z_ckpt/exp_dsr/checkpoints/step=02400.lora_only.ckpt
START=1
END=0

python3 \
inference.py \
$GPU \
./eval_csv/eval30.csv \
30 \
$CKPT $START $END \
