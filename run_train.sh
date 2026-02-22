CKPT_ROOT="$(realpath ../z_ckpt)"
TMP_ON_BIGDISK="${CKPT_ROOT}/.tmp"
mkdir -p "$TMP_ON_BIGDISK"
export TMPDIR="$TMP_ON_BIGDISK"
export TEMP="$TMP_ON_BIGDISK"
export TMP="$TMP_ON_BIGDISK"
export CKPT_TMP="${CKPT_ROOT}/.ckpt_tmp"
mkdir -p "$CKPT_TMP"


export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1


torchrun --nproc_per_node=4 \
  train.py \
  --task train \
  --train_architecture lora \
  --output_path ../ \
  --dit_path "../Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 48 \
  --max_epochs -1 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --lora_target_modules "text_embedding.0,text_embedding.2,cross_attn.k,cross_attn.v" \
  --use_gradient_checkpointing \
  --accumulate_grad_batches 12 \
  --train_batch_size 1 \
  --dataset_path "../data/tensors" \
  --metadata "../data/train_metadata.csv" \
  --expname exp_dsr
