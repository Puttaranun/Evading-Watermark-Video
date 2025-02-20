import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging
logging.getLogger("matplotlib.image").setLevel(logging.ERROR)
from IPython.display import HTML, display

import pandas as pd
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms.functional as F

import videoseal
from videoseal.augmentation import H264
from videoseal.evals.metrics import bit_accuracy
import argparse
import cv2
import numpy as np


def final_message_extractor(msg_path, video_path, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt_msgs = torch.tensor(np.loadtxt(msg_path))

    video_w, _, _ = torchvision.io.read_video(video_path, output_format="TCHW")
    video_w = video_w.to(device)
    video_w = video_w / 255.0
    video_w = video_w[:50]
    with torch.no_grad():
        msg_extracted = model.extract_message(video_w).cpu().detach()
        bit_accuracy_ = bit_accuracy(msg_extracted, gt_msgs).nanmean().item()
        print(f"Bit Accuracy: {bit_accuracy_:.3f}")
        print("GT:", gt_msgs[0])
        print("Extracted", msg_extracted.int())

