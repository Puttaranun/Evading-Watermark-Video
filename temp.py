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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./assets/videos/1.mp4')
    parser.add_argument('--output_path', type=str, default='./assets/videos/1_w.mp4')
    args = parser.parse_args()
    return args

def save_torch_video(video_w, out_path, fps):
    numpy_vid = torch.permute(video_w, (0, 2, 3, 1)).detach().numpy()
    numpy_vid = (numpy_vid * 255).astype(np.uint8)

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (numpy_vid.shape[2], numpy_vid.shape[1]))
    cv2.imwrite("test_frame.png", numpy_vid[0])
    for i in range(numpy_vid.shape[0]):
        numpy_vid[i] = cv2.cvtColor(numpy_vid[i], cv2.COLOR_BGR2RGB)
        out.write(numpy_vid[i])

    out.release()

def pgd_attack(model, img, gt_labels, alpha=0.007, eps=0.03, num_iter=20):
    bce_loss = torch.nn.BCEWithLogitsLoss()
    img_original = img.clone()
    for i in range(num_iter):
        img.retain_grad()
        detection = model.detect(torch.unsqueeze(img, dim=0), is_video=False)['preds'] 
    
        total_ce_loss = bce_loss(torch.squeeze(detection)[1:], torch.squeeze(1 - gt_labels))
        total_ce_loss.backward(retain_graph=True)
        with torch.no_grad():
            grad_sign = torch.sign(img.grad)
            img += alpha * grad_sign
            img = torch.clamp(img, min=img_original - eps, max=img_original + eps)
        img.requires_grad = True
        img.grad = None
        model.grad = None
        total_ce_loss = 0
        torch.cuda.empty_cache()
    return img

# def batched_pgd_attack(model, imgs, gt_labels, alpha=0, eps=0.001, num_iter=20):
#     bce_loss = torch.nn.BCEWithLogitsLoss()
#     imgs_original = imgs.clone()
#     for i in range(num_iter):
#         imgs.retain_grad()
#         detection = model.detect(imgs, is_video=False)['preds'] 
#         total_ce_loss = bce_loss(torch.squeeze(detection)[1:], torch.squeeze(1 - gt_labels))
#         total_ce_loss.backward(retain_graph=True)
#         grad_sign = torch.sign(imgs.grad)
#         with torch.no_grad():
#             imgs += alpha * grad_sign
#         imgs = torch.clip(imgs, imgs_original - eps, imgs_original + eps)
#         imgs.grad = None
#         model.zero_grad()
#     return imgs







def main(args):
    
    video_w, _, _ = torchvision.io.read_video('./assets/videos/1_w.mp4', output_format="TCHW")
    video_w_pgd, _, _  = = torchvision.io.read_video('./assets/videos/1_w_pgd.mp4', output_format="TCHW")
    
    pgd_vid = []
    #msg_extracted = model.extract_message(video_w)
    for i in range(0, 50):
        print("Attacking frame:", i)
        pgd_vid.append(pgd_attack(model, video_w[i], gt_labels=gt_msgs).cpu().detach())
        #pgd_vid.append(video_w[i].cpu().detach())
    pgd_vid = torch.stack(pgd_vid, dim=0)
    save_torch_video(pgd_vid, './assets/videos/1_w_pgd.mp4', fps)


    # bit_accuracy_ = bit_accuracy(msg_extracted, gt_msgs).nanmean().item()
    # print(f"Bit Accuracy: {bit_accuracy_:.3f}")
    # print("GT:", gt_msgs[0])
    # print("Extracted", msg_extracted.int())

if __name__ == '__main__':
    main(get_args())
    
