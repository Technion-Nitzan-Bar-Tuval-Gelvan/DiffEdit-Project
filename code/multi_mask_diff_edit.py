
import os
import numpy as np

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from fastcore.all import concat
from fastai.basics import show_image,show_images
from fastdownload import FastDownload
from pathlib import Path

from PIL import Image
import torch, logging
from torch import autocast
from torchvision import transforms as tfms

from huggingface_hub import notebook_login
from transformers import CLIPTextModel,CLIPTokenizer
from transformers import logging
from diffusers import AutoencoderKL,UNet2DConditionModel,LMSDiscreteScheduler,StableDiffusionInpaintPipeline
import cv2
from PIL import Image

#ours
from diff_edit import DiffEdit

def save_images_as_grid(images_list, grid_size, output_path):
    # Calculate the total number of images and ensure it matches the grid size
    total_images = len(images_list)
    if total_images != grid_size[0] * grid_size[1]:
        raise ValueError("Number of images doesn't match the grid size")
 
    # Get the size of each image
    image_width, image_height = images_list[0].size
 
    # Create a new image with the size of the grid
    grid_width = image_width * grid_size[0]
    grid_height = image_height * grid_size[1]
    grid_image = Image.new('RGB', (grid_width, grid_height))
 
    # Paste each image onto the grid
    for i, img in enumerate(images_list):
        row = i // grid_size[0]
        col = i % grid_size[0]
        grid_image.paste(img, (col * image_width, row * image_height))
 
    # Save the grid image
    grid_image.save(output_path)


# Function to dilate the masks
def morph_proc_masks(masks, kernel_size=1, iterations=1):
    processed_masks = []
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    for mask in masks:
        mask = np.array(mask)
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)
        eroded_mask = cv2.erode(dilated_mask.astype(np.uint8), kernel, iterations=iterations)
        proc_mask = Image.fromarray(eroded_mask.astype(np.uint8))
        processed_masks.append(proc_mask)
    return processed_masks


class MultiMaskDiffEdit(DiffEdit):
    def __init__(self, device):
        super().__init__(device)


    def multiple_masks_diffedit(self, im_path, prompts, generate_output, with_subtract, with_morphological, **kwargs):
        im_path = Path(im_path)
        out = []
        masks_list = []
        diff_masks_list = []
        binary_masks_list = []

        im = Image.open(im_path).resize((512,512))
        im_latent = self.image2latent(im)
        out.append(im)

        # Generate the diff mask for each pair of prompts
        for prompt in prompts:
            ref_prompt, query_prompt = prompt
            if 'seed' not in kwargs: kwargs['seed'] = torch.seed()
            diff_mask = self.calc_diffedit_diff(im_latent,ref_prompt,query_prompt,**kwargs)
            out.append(diff_mask.resize((512,512)))
            masks_list.append(diff_mask)
        
        # calculate the difference between masks
        if with_subtract:
            threshold = 100
            for i in range(len(masks_list)):
                diff = np.array(masks_list[i], dtype=np.float32)
                for j in range(i+1, len(masks_list)):
                    below_threshold = (np.array(masks_list[i], dtype=np.float32) < threshold)
                    diff = diff - np.array(masks_list[j], dtype=np.float32)
                    # normalize values between [-255, 255] to [0,255]
                    diff = ((diff - diff.min()) / (diff.max() - diff.min())) * 255
                    diff[below_threshold] = np.array(masks_list[i], dtype=np.float32)[below_threshold]
                diff_masks_list.append(Image.fromarray(diff.astype(np.uint8)))
        else:
            diff_masks_list = masks_list

        if with_morphological:
            diff_masks_list = morph_proc_masks(diff_masks_list, kernel_size=2, iterations=1)

        for i, mask in enumerate(diff_masks_list):
            out.append(mask.resize((512,512)))
            mask = self.process_diffedit_mask(mask, threshold=0.45, **kwargs) # binarize
            mask = mask.resize((512,512))
            out.append(mask)

            # save masks to files
            os.makedirs('masks', exist_ok=True)
            mask.save(f"masks/mask_{i}.jpg")
            binary_masks_list.append(mask)
    
        blended_img = im
        for prompt, mask in zip(prompts, binary_masks_list):
            ref_prompt, query_prompt = prompt
            out.append(mask)
            if generate_output:
                out.append(self.get_blended_mask(blended_img, mask))
                blended_img = self.inpaint(prompt=[query_prompt], image=blended_img, mask_image=mask,
                    generator=torch.Generator(self.device).manual_seed(kwargs['seed'])).images[0]
                out.append(blended_img)
            
        show_images(out)
        save_images_as_grid(out, (len(out), 1), 'results/output_grid_image.jpg')