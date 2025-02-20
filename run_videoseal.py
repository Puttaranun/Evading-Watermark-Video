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
    parser.add_argument('--input_path', type=str, default='1.mp4')
    parser.add_argument('--output_path', type=str, default='1_w.mp4')
    parser.add_argument('--pgd_output_path', type=str, default='1_w_pgd.mp4')

    args = parser.parse_args()
    args.input_path = './assets/videos/' + args.input_path
    args.output_path = './assets/videos/' + args.output_path
    args.pgd_output_path = './assets/videos/' + args.pgd_output_path
    
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

def pgd_attack(model, img, target_labels, alpha=0.007, eps=0.03, num_iter=20):
    bce_loss = torch.nn.BCEWithLogitsLoss()
    img_original = img.clone()
    for i in range(num_iter):
        img.retain_grad()
        detection = model.detect(torch.unsqueeze(img, dim=0), is_video=False)['preds'] 
        total_ce_loss = bce_loss(torch.squeeze(detection)[1:], torch.squeeze(target_labels.float()))
        total_ce_loss.backward(retain_graph=True)
        with torch.no_grad():
            grad_sign = torch.sign(img.grad)
            img -= alpha * grad_sign
            img = torch.clamp(img, min=img_original - eps, max=img_original + eps)
            img = torch.clamp(img, min=0, max=1)
        img.requires_grad = True
        img.grad = None
        model.grad = None
        total_ce_loss = 0
        torch.cuda.empty_cache()
    print((detection > 0).int())
    return img.detach().cpu()

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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fps = 24 

    # Load the VideoSeal model
    model = videoseal.load("videoseal")
    model.chunk_size=2

    # Set the model to evaluation mode and move it to the selected device
    model = model.eval()
    model = model.to(device)

    # # Path to the input video
    video_path = args.input_path
    # Read the video and convert to tensor format
    video, _, _ = torchvision.io.read_video(video_path, output_format="TCHW")

    # Normalize the video frames to the range [0, 1] and trim to 1 second
    video = video.float() / 255.0
    video = video.to(device)
    # Perform watermark embedding
    gt_msgs = torch.zeros(1, 96)
    with torch.no_grad():
        outputs = model.embed(video, is_video=True, msgs=gt_msgs.to(device))

    # Extract the results
    video_w = outputs["imgs_w"].cpu().detach()  # Watermarked video frames

    save_torch_video(video_w, args.output_path, fps)

    np.savetxt('./assets/videos/1_msgs.txt', gt_msgs, fmt='%d')

    video_w, _, _ = torchvision.io.read_video(args.output_path, output_format="TCHW")
    video_w = video_w / 255.0
    #video_w = video_w[:, :, :400, :400]
    video_w = video_w.to(device)
    video_w.requires_grad = True

    # Get PGD attacked video
    video_pgd = []
    for i in range(video_w.shape[0]):
        #gt_labels = (torch.rand_like(gt_msgs) > 0).int()
        gt_labels = 1 - gt_msgs.to(device)
        video_pgd.append(pgd_attack(model, video_w[i], target_labels=gt_labels))
        torch.cuda.empty_cache()

 
    video_pgd = torch.stack(video_pgd, axis=0)
    save_torch_video(video_pgd, args.pgd_output_path, fps)

    with torch.no_grad():
        final_message_extractor('./assets/videos/1_msgs.txt', args.output_path, model)
        final_message_extractor('./assets/videos/1_msgs.txt', args.pgd_output_path, model)

if __name__ == '__main__':
    main(get_args())
    
