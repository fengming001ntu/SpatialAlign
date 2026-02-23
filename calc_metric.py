import pandas as pd
import pickle
import os
import torch
import numpy as np
import sys
import math
import torch.nn.functional as Func
import torchvision.transforms as Tran
import torch.hub
from torchvision.io import read_video
from typing import List, Optional, Union, Tuple
from tqdm import tqdm
from collections import defaultdict
from torchmetrics.multimodal import CLIPImageQualityAssessment

def average_scalar_by_key(list_of_dicts):
    values = defaultdict(list)

    for d in list_of_dicts:
        for k, v in d.items():
            values[k].append(v.mean())

    avg_dict = {k: torch.stack(vs).mean() for k, vs in values.items()}

    return avg_dict


DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD  = (0.229, 0.224, 0.225)



def dinov2_frame_transform(image_size: int = 518):
    return Tran.Compose([
        Tran.Resize(image_size, interpolation=Tran.InterpolationMode.BICUBIC, antialias=True),
        Tran.CenterCrop(image_size),
        Tran.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
    ])

def load_video_raw_frames(
    video_path: str,
    max_frames: int = 81,
    # frame_subsample: int = 2,
):
    video, _, info = read_video(video_path, pts_unit="sec")

    frames = video.permute(0, 3, 1, 2).float() / 255.0  # [T,3,H,W]
    if max_frames is not None and frames.shape[0] > max_frames:
        frames = frames[:max_frames]

    return [f for f in frames]   # list of [3,H,W]

def compute_motion_acc_smoothness(flows):
    acc_mags = []

    for k in range(len(flows) - 1):
        # [1,2,H,W] -> [2,H,W]
        v1 = flows[k][0]      # (2,H,W)
        v2 = flows[k+1][0]    # (2,H,W)

        ax = v2[0] - v1[0]    # [H,W]
        ay = v2[1] - v1[1]    # [H,W]

        a_mag = torch.sqrt(ax**2 + ay**2)
        acc_mags.append(a_mag.mean())

        print("----------------", k, "----------------")
        print(a_mag)

        v_mag = torch.sqrt(v1[0]**2 + v1[1]**2)
        print(v_mag)
        print()

    mean_acc = torch.stack(acc_mags).mean().item()
    smoothness = 1.0 / (1.0 + mean_acc)
    return float(smoothness)


#################################################################################
height = 480
width = 832

df = pd.read_csv(sys.argv[1])
img_dir = sys.argv[2]
det_dir = os.path.normpath(sys.argv[3])
depth_dir = det_dir.replace('_det', '_depth')

result_df = []

vis_dir = os.path.normpath(det_dir).replace('_det', '_scoreV5')
os.makedirs(vis_dir, exist_ok=True)

DEVICE = "cuda"

model_dinov2 = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitb14"
)
model_dinov2.to(DEVICE)
model_dinov2.eval()
proc_transform = dinov2_frame_transform()

clipiqa_prompt = ("natural", "quality")

model_clipiqa = CLIPImageQualityAssessment(
    data_range=1.0, 
    prompts=clipiqa_prompt,
).to(DEVICE)

result_id_consistency = []
result_smoothness = []

score_clipiqa_total = []

for kdx, row in tqdm(df.iterrows(), total=len(df)):
    prompt_id = row["prompt_id"]
    animal = row["animal"]
    static = row["object"]
    initial_spaword = row["initial_SR"]
    final_spaword = row["final_SR"]
    ref_labels = [animal, static]

    dsr_type = row["type"]
    initial_type = dsr_type[0]
    final_type = dsr_type[1]

    
    file_list = []
    aggregate_reward_list = []

    
    for sample_idx in range(5):
        video_name = f"{prompt_id}_F81_seed{sample_idx:02d}.mp4"

        file_path = os.path.join(det_dir, video_name+".pkl")
        if not os.path.exists(file_path):
            break

        frames = load_video_raw_frames(os.path.join(img_dir, video_name))

        ##### calc clipiqa ##########
        selected_idx = list(range(0,81,1))
        selected_frames = [frames[i].unsqueeze(0).to(DEVICE) for i in selected_idx]
        selected_frames = torch.cat(selected_frames, dim=0)
        score_clipiqa = model_clipiqa(selected_frames) #.mean().item()

        score_clipiqa_total.append(score_clipiqa)

        file_list.append(video_name)
        
        with open(file_path, 'rb') as f:
        	det_data = pickle.load(f)

        if len(det_data) == 0:
            aggregate_reward_list.append(-100)
            continue

        if ("B" in row["type"]) or ("F" in row["type"]):
            depth_total = np.load(os.path.join(depth_dir, video_name.replace(".mp4", "_depths.npy")))

            has_negative = np.any(depth_total < 0)
            if has_negative:
                print(video_name)
                print("disparity has negative")
                continue
        

        animal_img_batch = []
        flow_list = []

        initial_reward_list = []
        final_reward_list = []
        for det_frame in det_data:
            f_idx, det_labels, bbox = det_frame[:3]
            if len(det_frame) == 4:
                mask = det_frame[3]
            else:
                mask = None

            if sorted(ref_labels) == sorted(det_labels):
                order = [ref_labels.index(item) for item in det_labels]
                bbox = bbox[order]
                if mask is not None:
                    mask = mask[order]
            else:
                continue

            animal_bbox = bbox[0]
            static_bbox = bbox[1]

            animal_L, animal_T, animal_R, animal_B  = animal_bbox
            animal_center_w = 0.5 * (animal_L + animal_R)
            animal_center_h = 0.5 * (animal_T + animal_B)
            static_L, static_T, static_R, static_B  = static_bbox
            static_center_w = 0.5 * (static_L + static_R)
            static_center_h = 0.5 * (static_T + static_B)

            animal_w = animal_R - animal_L
            static_w = static_R - static_L
            width_max = 0.5 * (animal_w + static_w)
            animal_h = animal_B - animal_T
            static_h = static_B - static_T
            height_max = 0.5 * (animal_h + static_h)
            
            if (width_max == 0) or \
               (height_max == 0) or \
               (animal_w == 0) or \
               (static_w == 0):
                break

            delta_center_w = (animal_center_w - static_center_w) / width_max
            delta_center_h = (animal_center_h - static_center_h) / height_max

            height_max_under = min(0.1*animal_h, 0.1*static_h)
            delta_center_h_under = (animal_center_h - static_center_h) / height_max_under
            
            vector = np.array([animal_center_w - static_center_w, animal_center_h - static_center_h])

            x_ax = np.array([1,0])
            numerator_x = np.dot(vector, x_ax)
            denominator_x = np.linalg.norm(vector) * np.linalg.norm(x_ax)

            if denominator_x == 0:
                cos_theta_x = 0.0  # 或 np.nan, 或其他你需要的默认值
            else:
                cos_theta_x = (numerator_x / denominator_x).item()

            distance_x = np.clip( abs(delta_center_w), 0.0, 1.0).item()
            
            if initial_type == "L":
                reward = -cos_theta_x * distance_x
                initial_reward_list.append(reward)
            if initial_type == "R":
                reward =  cos_theta_x * distance_x
                initial_reward_list.append(reward)

            if final_type == "L":
                reward = -cos_theta_x * distance_x
                final_reward_list.append(reward)
            if final_type == "R":
                reward =  cos_theta_x * distance_x
                final_reward_list.append(reward)

            y_ax = np.array([0,1]) # pointing downwards
            numerator_y = np.dot(vector, y_ax)
            denominator_y = np.linalg.norm(vector) * np.linalg.norm(y_ax)

            if denominator_y == 0:
                cos_theta_y = 0.0
            else:
                cos_theta_y = (numerator_y / denominator_y).item()

            if initial_type == "T":
                reward = -cos_theta_y
                reward = reward * np.clip( abs(delta_center_h), 0.0, 1.0).item()
                initial_reward_list.append(reward)
            if initial_type == "U":
                reward =  cos_theta_y
                reward = reward * np.clip( abs(delta_center_h_under), 0.0, 1.0).item()
                initial_reward_list.append(reward)

            if final_type == "T":
                reward = -cos_theta_y
                reward = reward * np.clip( abs(delta_center_h), 0.0, 1.0).item()
                final_reward_list.append(reward)
            if final_type == "U":
                reward =  cos_theta_y
                reward = reward * np.clip( abs(delta_center_h_under), 0.0, 1.0).item()
                final_reward_list.append(reward)

            animal_img = frames[f_idx][:,animal_T:animal_B,animal_L:animal_R]
            animal_img = proc_transform(animal_img)
            animal_img_batch.append(animal_img)

        if len(initial_reward_list) >= 20:
            pass

            ##### calc ID consistency ##########
            animal_img_batch = torch.stack(animal_img_batch, dim=0).to(DEVICE)

            with torch.no_grad():
                animal_feat = model_dinov2(animal_img_batch)
                animal_feat = Func.normalize(animal_feat, dim=1)
            # print(animal_feat.shape)

            F_valid = animal_feat.shape[0]
            
            ref_feat = animal_feat[0:1]
            # print(animal_feat.shape)
            pre_feat = animal_feat[0:F_valid-1]
            cur_feat = animal_feat[1:F_valid]

            sims_to_ref = (cur_feat @ ref_feat.T).squeeze(1)
            sims_to_pre = (pre_feat * cur_feat).sum(dim=1)
            sims_sum = (sims_to_ref + sims_to_pre) / 2

            score_id_consistency = float(sims_sum.mean().item())
            result_id_consistency.append(score_id_consistency)


################################################################
print()
print(img_dir)
metric_id_consistency = sum(result_id_consistency) / len(result_id_consistency)
print(metric_id_consistency)

result = average_scalar_by_key(score_clipiqa_total)

print(result)