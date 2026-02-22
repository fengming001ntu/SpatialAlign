python3 calc_dsr_score.py \
  ./eval_csv/eval30.csv \
  ../z_ckpt/exp_dsr/checkpoints/step=02400.lora_only.ckpt_eval_30_det

python3 calc_correctness.py \
  ../z_ckpt/exp_dsr/checkpoints/result_DSRscoreV5_step=02400.lora_only.ckpt_eval_30_det.csv
