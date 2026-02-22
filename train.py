# --- put this block at the VERY TOP of the file (before any other imports) ---
import os, pathlib, tempfile
# Some Python builds need this import name for patching; ignore if not present
try:
    import multiprocessing.util as _mp_util
except Exception:
    _mp_util = None

def _set_per_rank_tmpdir_early():
    # Base priority: TMPDIR/TMP/TEMP/CKPT_TMP/TMP_BASE or /tmp
    base = (os.environ.get("TMPDIR") or os.environ.get("TMP") or os.environ.get("TEMP")
            or os.environ.get("CKPT_TMP") or os.environ.get("TMP_BASE") or "/tmp")
    lr = os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or "0"
    per = os.path.join(base, f"rank{lr}")
    pathlib.Path(per).mkdir(parents=True, exist_ok=True)

    # 1) Env for children
    os.environ["TMPDIR"] = per
    os.environ["TMP"]    = per
    os.environ["TEMP"]   = per

    # 2) Force Python's temp modules to use the per-rank dir
    tempfile.tempdir = per          # affects future tempfile.gettempdir()
    tempfile.gettempdir()           # materialize

    # 3) If multiprocessing already grabbed a tempdir, override it
    if _mp_util is not None and hasattr(_mp_util, "_tempdir"):
        _mp_util._tempdir = per     # used by multiprocessing finalizers

_set_per_rank_tmpdir_early()
# -------------------------------------------------------------------------------

import torch.distributed as dist

import torch, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np

import pickle
from lightning.pytorch.loggers import TensorBoardLogger
import copy
from torch.optim.lr_scheduler import LambdaLR
import math
import ast
import random

import signal
import threading

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch):
        def process_reward(x_in):
            y = []
            for x in x_in: 
                if x >= -1 and x < 0.7:
                    y.append(0)
                elif x >= 0.7:
                    y.append(1)
                else:
                    y.append(-100)
            return y

        self.metadata = pd.read_csv(metadata_path)
        self.metadata["reward"] = self.metadata["reward"].apply(lambda x: ast.literal_eval(x))
        self.metadata["valid"] = self.metadata["reward"].apply(lambda x: len(x) - x.count(-100))
        self.metadata = self.metadata[self.metadata["valid"]>=2] # at least two samples are not bad

        self.metadata["reward"] = self.metadata["reward"].apply(lambda x: process_reward(x))
        self.metadata["valid"] = self.metadata["reward"].apply(lambda x: x.count(1))
        self.metadata = self.metadata[self.metadata["valid"]>0]
        self.metadata["valid"] = self.metadata["reward"].apply(lambda x: x.count(0))
        self.metadata = self.metadata[self.metadata["valid"]>0] # at least two samples are not bad

        

        index_list = self.metadata["prompt_id"].to_list()
        file_name_list = [os.path.join(base_path, f"{x}.tensors.pth") for x in index_list]
        self.path = [item for item in file_name_list if os.path.exists(item)]
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0
        # exit()
        
        self.steps_per_epoch = steps_per_epoch


    def __getitem__(self, idx):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + idx) % len(self.path) # For fixed seed.
        path = self.path[data_id]
        data = torch.load(path, weights_only=True, map_location="cpu")

        index = data["prompt_id"]
        row = self.metadata[self.metadata["prompt_id"] == index].to_dict(orient='records')[0]
        data["reward"] = np.asarray(row["reward"]).astype(np.float16)

        return data
    

    def __len__(self):
        return self.steps_per_epoch
        # return len(self.path)

class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()

        self.ref_model = copy.deepcopy(self.pipe.denoising_model())
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval()

        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        self.my_interrupted = False
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    

    def training_step(self, batch, batch_idx):
        # 1. Load latents and prepare timestep
        self.pipe.device = self.device

        ins_index = batch["prompt_id"]
        latents_total = batch["latents_data"] #.to(self.device)
        reward_total = batch["reward"]
        B, N, C, noF, H, W = latents_total.shape

        selected = np.zeros((B, 2), dtype=int)
        for bix in range(B):
            row_valid = np.arange(0,N)
            idx_pos, idx_neg = 0, 0
            while (reward_total[bix, idx_pos] != 1) or (reward_total[bix, idx_neg] != 0):

                idx_pos, idx_neg = np.random.choice(row_valid, size=2, replace=False)

            selected[bix] = [idx_pos, idx_neg]

        batch_idx = np.arange(B)[:, None]
        latents = latents_total[batch_idx, selected]
        latents_pos, latents_neg = latents[:,0], latents[:,1]


        noise_pos = torch.randn_like(latents_pos)
        noise_neg = torch.randn_like(latents_neg)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)


        noisy_pos = self.pipe.scheduler.add_noise(latents_pos, noise_pos, timestep)
        noisy_neg = self.pipe.scheduler.add_noise(latents_neg, noise_neg, timestep)


        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"].squeeze(1).to(self.device)
        extra_input = self.pipe.prepare_extra_input(latents_pos)



        # 2. Current model prediction
        pred_pos = self.pipe.denoising_model()(
            noisy_pos, timestep=timestep, **prompt_emb, **extra_input,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )

        pred_neg = self.pipe.denoising_model()(
            noisy_neg, timestep=timestep, **prompt_emb, **extra_input,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )


        # 3. Reference model prediction
        with torch.no_grad():
            pred_pos_ref = self.ref_model(
                noisy_pos, timestep=timestep, **prompt_emb, **extra_input,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
            )

            pred_neg_ref = self.ref_model(
                noisy_neg, timestep=timestep, **prompt_emb, **extra_input,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
            )

        target_pos = self.pipe.scheduler.training_target(latents_pos, noise_pos, timestep)
        target_neg = self.pipe.scheduler.training_target(latents_neg, noise_neg, timestep)


        # 4. MSE-based scores (batchwise)
        mse_pos = torch.nn.functional.mse_loss(pred_pos.float(), target_pos.float(), reduction='none').mean(dim=[1,2,3,4])
        mse_neg = torch.nn.functional.mse_loss(pred_neg.float(), target_neg.float(), reduction='none').mean(dim=[1,2,3,4])
        mse_pos_ref = torch.nn.functional.mse_loss(pred_pos_ref.float(), target_pos.float(), reduction='none').mean(dim=[1,2,3,4])
        mse_neg_ref = torch.nn.functional.mse_loss(pred_neg_ref.float(), target_neg.float(), reduction='none').mean(dim=[1,2,3,4])

        score_pos = mse_pos_ref - mse_pos
        score_neg = mse_neg_ref - mse_neg


        # 5. DPO loss
        beta = 1.0
        margin = beta * (score_pos - score_neg)
        margin = torch.clip(margin, min=-5.0, max=5.0)
        dpo_loss = -torch.log(torch.sigmoid(margin)).mean()

        # 6. Regularizer
        reg_loss = 0.5 * (
            torch.nn.functional.mse_loss(pred_pos.float(), pred_pos_ref.float()) +
            torch.nn.functional.mse_loss(pred_neg.float(), pred_neg_ref.float())
        )

        lambda_reg = 0.5
        total_loss = dpo_loss + lambda_reg * reg_loss

        # 7. Logging
        self.log("train/loss_total", total_loss, prog_bar=False)
        self.log("train/loss_dpo", dpo_loss, prog_bar=False)
        self.log("train/loss_reg", ref_loss, prog_bar=False)
        self.log("train/score_pos", score_pos.mean(), prog_bar=False)
        self.log("train/score_neg", score_neg.mean(), prog_bar=False)
        self.log("train/score_margin", (score_pos - score_neg).mean(), prog_bar=False)

        return total_loss



    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)

        def lr_lambda(current_step):
            warmup_steps = 300
            total_steps = 3000

            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    

    def on_save_checkpoint(self, checkpoint):
        print(self.my_interrupted)
        # if not self.my_interrupted:
        #     checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        print("on_save_checkpoint(): saving checkpoint")
        checkpoint.update(lora_state_dict)
        print("on_save_checkpoint(): saving checkpoint finished")

class SaveOnInterruptCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.interrupted = False

        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

    def on_train_start(self, trainer, pl_module):
        self.save_dir = trainer.logger.log_dir
        self.flag_path = os.path.join(self.save_dir, "signalfile.txt")
        trainer.my_interrupted = False

    def handle_signal(self, signum, frame):
        self.interrupted = True
        print(f"\n[Interrupt] Signal {signum} received. Will save full checkpoint at end of current step...")

    def on_exception(self, trainer, pl_module, exception):
        pl_module.my_interrupted = True
        step = trainer.global_step
        print(f"\n[Exception] Saving full checkpoint")
        if getattr(trainer, "is_global_zero", True):
            trainer.save_checkpoint(os.path.join(self.save_dir, f"step={step:05d}.ckpt"))
        trainer.should_stop = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        is_rank0 = getattr(trainer, "is_global_zero", True)


        if self.interrupted:
            pl_module.my_interrupted = True
            if is_rank0:
                step = trainer.global_step
                print(f"\n[Interrupt] Saving full checkpoint (step={step})")
                trainer.save_checkpoint(os.path.join(self.save_dir, f"step={step:05d}.ckpt"))
            trainer.should_stop = True
            return

        if os.path.exists(self.flag_path):
            pl_module.my_interrupted = True
            step = trainer.global_step


            if is_rank0:
                print(f"\n[Watcher] Saving full checkpoint (step={step})")


            trainer.save_checkpoint(os.path.join(self.save_dir, f"step={step:05d}.ckpt"))

            if is_rank0:
                try:
                    os.remove(self.flag_path)
                except FileNotFoundError:
                    pass

            trainer.should_stop = True
            return


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--ckpt_resume",
        type=str,
        default=None,
        required=False,
        help="The checkpoint for resume training.",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Number of steps per epoch.",
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        required=True,
        help="The path of the medata csv file.",
    )

    parser.add_argument(
        "--expname",
        type=str,
        default="test",
        required=False,
        help="The experiment name.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    args = parser.parse_args()
    return args
    
def train(args):
    dataset = TensorDataset(
        args.dataset_path,
        args.metadata,
        steps_per_epoch=args.steps_per_epoch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
    )

    logger = TensorBoardLogger(args.output_path, name="z_ckpt_DSR", version=args.expname)
    logger.log_hyperparams(params=args)
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[
            pl.pytorch.callbacks.ModelCheckpoint(
                save_top_k=-1,
                every_n_train_steps=100,
                save_on_train_epoch_end=False,
                filename="{step:05d}",
            ),
            SaveOnInterruptCallback()
        ],
        logger=logger,
        log_every_n_steps=1,
    )

    
    logger.log_hyperparams(params={"num_devices": trainer.num_devices})

    if args.ckpt_resume is not None:
        trainer.fit(model, dataloader, ckpt_path=args.ckpt_resume)
    else:
        trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    train(args)
