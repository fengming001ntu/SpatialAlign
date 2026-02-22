import os
import sys

GPU=sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = GPU


import cv2
import torch
import numpy as np
import supervision as sv

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
# from utils.track_utils import sample_points_from_masks
# from utils.video_utils import create_video_from_images

import subprocess
import pickle
import pandas as pd
import imageio

def create_video_from_images(image_folder, output_video_path, frame_rate=25):
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    
    # get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()  # sort the files in alphabetical order
    # print(image_files)

    writer = imageio.get_writer(output_video_path, fps=16, quality=5)
    for item in tqdm(image_files):
        frame = imageio.imread(os.path.join(image_folder, item))
        writer.append_data(frame)
    writer.close()

    # if not image_files:
    #     raise ValueError("No valid image files found in the specified folder.")
    
    # # load the first image to get the dimensions of the video
    # first_image_path = os.path.join(image_folder, image_files[0])
    # first_image = cv2.imread(first_image_path)
    # height, width, _ = first_image.shape
    
    # # create a video writer
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for saving the video
    # video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    # write each image to the video
    # for image_file in tqdm(image_files):
    #     image_path = os.path.join(image_folder, image_file)
    #     image = cv2.imread(image_path)
    #     video_writer.write(image)
    
    # # source release
    # video_writer.release()
    # print(f"Video saved at {output_video_path}")


"""
Hyperparam for Ground and Tracking
"""
MODEL_ID = "IDEA-Research/grounding-dino-tiny"

"""
Step 1: Environment settings and model initialization for SAM 2
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "../../Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

print("Loading models")
video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# build grounding dino from huggingface
model_id = MODEL_ID
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


# print(type(processor))
# exit()
# VIDEO_PATH = "./assets/hippopotamus.mp4"
# TEXT_PROMPT = "hippopotamus."
# OUTPUT_VIDEO_PATH = "./hippopotamus_tracking_demo.mp4"
# SOURCE_VIDEO_FRAME_DIR = "../tmp_detection_use"
# SAVE_TRACKING_RESULTS_DIR = "./tracking_results"
PROMPT_TYPE_FOR_VIDEO = "box" # choose from ["point", "box", "mask"]


import secrets
import string

def generate_random_string_secrets(length):
   """使用secrets模块生成随机字符串"""
   alphabet = string.ascii_letters + string.digits
   return ''.join(secrets.choice(alphabet) for i in range(length))


def first_max_run_ge4(nums):
    start = 0
    for i in range(1, len(nums) + 1):
        # 碰到断裂或结束
        if i == len(nums) or nums[i] != nums[i-1] + 1:
            run = nums[start:i]
            if len(run) >= 4:
                return run
            start = i
    return []

###############################################################

SOURCE_VIDEO_FRAME_DIR = f"../tmp_detection_use/{generate_random_string_secrets(8)}"
SAVE_TRACKING_RESULTS_DIR = SOURCE_VIDEO_FRAME_DIR + '_track'
os.makedirs(SAVE_TRACKING_RESULTS_DIR, exist_ok=True)

data_name = sys.argv[2]

df = pd.read_csv(sys.argv[3])
img_dir = sys.argv[4]

det_dir = os.path.normpath(img_dir) + '_det'
os.makedirs(det_dir, exist_ok=True)

tracked_video_dir = os.path.normpath(img_dir) + '_track'
os.makedirs(tracked_video_dir, exist_ok=True)

interval = int(sys.argv[5])
remainder = int(sys.argv[6])


for kdx, row in df.iterrows():
    prompt = row["prompt"]
    # prompt = row["prompt_from_to"]
    prompt_id = row["prompt_id"]
    if not (kdx % interval == remainder):
        continue

    NUM_SAMPLE = 10
    for sample_idx in range(NUM_SAMPLE):
        # if not (sample_idx % interval == remainder):
        #     continue
        print()
        print(prompt_id, "seed", sample_idx)

        file_name = f"{prompt_id}_F81_{data_name}{sample_idx:02d}.mp4"
        VIDEO_PATH = os.path.join(img_dir, file_name)
        if not os.path.exists(VIDEO_PATH):
            continue

        TEXT_PROMPT = f"{row['animal']}. {row['object']}."
        ref_labels = [row['animal'], row['object']]
        OUTPUT_VIDEO_PATH = os.path.join(tracked_video_dir, file_name)


        det_file = os.path.join(det_dir, os.path.basename(VIDEO_PATH)+'.pkl')
        if os.path.exists(det_file):
            continue

        """
        Custom video input directly using video files
        """
        video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # get video info
        # print(video_info)
        frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

        # saving video to frames
        source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
        source_frames.mkdir(parents=True, exist_ok=True)

        with sv.ImageSink(
            target_dir_path=source_frames, 
            overwrite=True, 
            image_name_pattern="{:05d}.jpg"
        ) as sink:
            # for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            for frame in frame_generator:
                sink.save_image(frame)

        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # frame_idx = 0  # the frame index we interact with

        
        det_total = []
        for frame_idx in range(0,81):
            # print("Starting frame:", frame_idx)

            """
            Step 2: Prompt Grounding DINO 1.5 with Cloud API for box coordinates
            """

            # prompt grounding dino to get the box coordinates on specific frame
            img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx])
            image = Image.open(img_path)
            inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = grounding_model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )

            input_boxes = results[0]["boxes"].cpu().numpy()
            confidences = results[0]["scores"].cpu().numpy().tolist()
            class_names = results[0]["labels"]

            det_total.append([frame_idx, input_boxes, confidences, class_names])
            print(frame_idx, class_names, confidences)


        stat_miss = 0
        stat_redundant = 0
        stat_valid = 0

        f_idx_valid = []

        for [frame_idx, input_boxes, confidences, class_names] in det_total:
            # print(frame_idx, class_names)
            if len(class_names) < 2:
                stat_miss += 1
            elif len(class_names) > 2:
                stat_redundant += 1
            else:
                if (sorted(ref_labels) == sorted(class_names)):
                    stat_valid +=1
                    f_idx_valid.append(frame_idx)
                else:
                    stat_redundant += 1

        print(stat_valid, stat_miss, stat_redundant)
        print(f_idx_valid)

        if (stat_redundant > 16) or (stat_valid < 20):
            det_data = []
            with open(det_file, 'wb') as f:
                pickle.dump(det_data, f)
            continue
        else:
            prop_main = first_max_run_ge4(f_idx_valid)
        
            if len(prop_main) < 4:
                det_data = []
                with open(det_file, 'wb') as f:
                    pickle.dump(det_data, f)
                continue

        ############################# CHECK LINE #######################################
        prop_setting = [(prop_main[0], False), (prop_main[0], True)]
        det_data = []
        for (prop_idx, prop_reverse) in prop_setting:
            print(prop_idx, prop_reverse)

            det_item = next((x for x in det_total if x[0] == prop_idx), None)
            frame_idx, input_boxes, confidences, OBJECTS = det_item

            img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[prop_idx])
            image = Image.open(img_path)
            image_predictor.set_image(np.array(image.convert("RGB")))

            # init video predictor state
            inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)


            # prompt SAM 2 image predictor to get the mask for the object
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            # convert the mask shape to (n, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)

            """
            Step 3: Register each object's positive points to video predictor with seperate add_new_points call
            """
            for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
                _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=prop_idx,
                    obj_id=object_id,
                    box=box,
                )


            """
            Step 4: Propagate the video predictor to get the segmentation results for each frame
            """
            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, reverse=prop_reverse):
                # print(out_frame_idx, out_obj_ids, out_mask_logits.shape)
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            """
            Step 5: Visualize the segment results across the video and save them
            """
            ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
            print(ID_TO_OBJECTS)
            for frame_idx, segments in video_segments.items():
                img = cv2.imread(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx]))
                
                object_ids = list(segments.keys())
                masks = list(segments.values())
                masks = np.concatenate(masks, axis=0)
                
                detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                    mask=masks, # (n, h, w)
                    class_id=np.array(object_ids, dtype=np.int32),
                )

                # print(frame_idx, object_ids, detections.xyxy)
                obj_names = [ID_TO_OBJECTS[j] for j in object_ids]

                if ("B" in row["type"]) or ("F" in row["type"]):
                    det_data.append([frame_idx, obj_names, detections.xyxy, masks])
                else:
                    det_data.append([frame_idx, obj_names, detections.xyxy])

                box_annotator = sv.BoxAnnotator()
                annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
                label_annotator = sv.LabelAnnotator()
                annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
                mask_annotator = sv.MaskAnnotator()
                annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
                cv2.imwrite(os.path.join(SAVE_TRACKING_RESULTS_DIR, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)


        """
        Step 6: Convert the annotated frames to video
        """
        with os.scandir(SAVE_TRACKING_RESULTS_DIR) as entries:
            if not any(entries):
                print("文件夹为空")
            else:
                print("文件夹不为空")
                create_video_from_images(SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH)

        det_data = list({x[0]: x for x in det_data}.values())
        # print(det_data)
        with open(det_file, 'wb') as f:
            pickle.dump(det_data, f)
        # exit()

subprocess.run(["rm", "-r", SOURCE_VIDEO_FRAME_DIR])
subprocess.run(["rm", "-r", SAVE_TRACKING_RESULTS_DIR])