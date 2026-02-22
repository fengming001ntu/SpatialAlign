import pandas as pd
import pickle
from tqdm import tqdm
import os
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
import math


height = 480
width = 832

df = pd.read_csv(sys.argv[1])
det_dir = os.path.normpath(sys.argv[2])


result_df = []

vis_dir = os.path.normpath(det_dir).replace('_det', '_scoreV5')
os.makedirs(vis_dir, exist_ok=True)

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
    
    
    for sample_idx in range(0,5):
        video_name = f"{prompt_id}_F81_seed{sample_idx:02d}.mp4"

        file_path = os.path.join(det_dir, video_name+".pkl")
        if not os.path.exists(file_path):
            continue

        file_list.append(video_name)
        
        with open(file_path, 'rb') as f:
        	det_data = pickle.load(f)

        if len(det_data) == 0:
            aggregate_reward_list.append(-100)
            continue

        initial_reward_list = []
        final_reward_list = []
        valid_f_idx = []

        det_data = sorted(det_data, key=lambda x: x[0])

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

            animal_L, animal_T, animal_R, animal_B  = bbox[0]
            animal_center_w = 0.5 * (animal_L + animal_R)
            animal_center_h = 0.5 * (animal_T + animal_B)
            static_L, static_T, static_R, static_B  = bbox[1]
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
                continue

            delta_center_w = (animal_center_w - static_center_w) / width_max
            delta_center_h = (animal_center_h - static_center_h) / height_max
            
            vector = np.array([animal_center_w - static_center_w, animal_center_h - static_center_h])

            x_ax = np.array([1,0])
            numerator_x = np.dot(vector, x_ax)
            denominator_x = np.linalg.norm(vector) * np.linalg.norm(x_ax)

            if denominator_x == 0:
                cos_theta_x = 0.0
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

            y_ax = np.array([0,1])
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

            if final_type == "T":
                reward = -cos_theta_y
                reward = reward * np.clip( abs(delta_center_h), 0.0, 1.0).item()
                final_reward_list.append(reward)

            valid_f_idx.append(f_idx)


        #####################################
        if len(initial_reward_list) >= 20:
            initial_reward_list = np.asarray(initial_reward_list)
            initial_SSR_score = initial_reward_list[:3].mean()
            initial_gap = 0.5*(initial_reward_list[0] - initial_reward_list[-1])

            final_reward_list = np.asarray(final_reward_list)
            final_SSR_score = final_reward_list[-3:].mean()
            final_gap = 0.5*(final_reward_list[-1] - final_reward_list[0])

            aggregate_reward = (initial_SSR_score + initial_gap + final_SSR_score + final_gap) / 8 + 0.5
            aggregate_reward_list.append(float(aggregate_reward))

            
            plt.figure(figsize=(6,8))
            plt.plot(valid_f_idx, initial_reward_list, label=f"initial: {initial_spaword}", linewidth=3)
            plt.plot(valid_f_idx, final_reward_list, label=f"final: {final_spaword}", linewidth=3)
            plt.xlim(0,81)
            plt.ylim(-1,1)
            plt.xlabel("Frame Index")
            plt.ylabel("Score")
            plt.title(
                f'''
                initial SSR score: {initial_SSR_score:.3f}
                initial gap      : {initial_gap:.3f}
                final   SSR score: {final_SSR_score:.3f}
                final   gap      : {final_gap:.3f}
                ''',
                fontsize=20, 
            )
            plt.legend(
                fontsize=24,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),
                # ncol=2,
                frameon=False
            )
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, video_name+'.png'), bbox_inches='tight', pad_inches=0.1)
            plt.close()


        else:
            aggregate_reward_list.append(-100)


    result_check = row.copy()
    result_check['video'] = file_list
    result_check['reward'] = aggregate_reward_list
    result_df.append(result_check)

result_df = pd.DataFrame(result_df)
parent_dir = os.path.dirname(det_dir)
base_name = os.path.basename(det_dir)
result_df.to_csv(os.path.join(parent_dir, f"result_DSRscoreV5_{base_name}.csv"), index=False)

