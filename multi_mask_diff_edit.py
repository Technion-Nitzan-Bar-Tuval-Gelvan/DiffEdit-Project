
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
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)
        eroded_mask = cv2.erode(dilated_mask.astype(np.uint8), kernel, iterations=iterations)
        proc_mask = Image.fromarray(eroded_mask.astype(np.uint8))
        processed_masks.append(proc_mask)
    return processed_masks

# class MultiMaskDiffEdit(DiffEdit):
#     def __init__(self, device):
#         super().__init__(device)


#     def multiple_masks_diffedit(self, im_path, prompts, generate_output, **kwargs):
#         im_path = Path(im_path)
#         out = []
#         masks_list = []
#         binary_masks_list = [] # List of binary masks

#         im = Image.open(im_path).resize((512,512))
#         im_latent = self.image2latent(im)
#         out.append(im)

#         # Generate the diff mask for each pair of prompts
#         for prompt in prompts:
#             ref_prompt, query_prompt = prompt
#             if 'seed' not in kwargs: kwargs['seed'] = torch.seed()
#             diff_mask = self.calc_diffedit_diff(im_latent,ref_prompt,query_prompt,**kwargs)
#             out.append(diff_mask.resize((512,512)))
#             masks_list.append(diff_mask)

#         # Combine the masks into a single mask
#         combined_mask = np.zeros_like(masks_list[0])
        
#         # Iterate through each mask and update the combined mask with maximum values
#         for mask in masks_list:
#             combined_mask = np.maximum(combined_mask, mask)
        
#         # Iterate through each mask and set non-maximum pixels to zero
#         for i, mask in enumerate(masks_list):
#             mask = np.array(mask)
#             masks_list[i] = np.where(mask != combined_mask, 0, mask)
#             # masks_list[i] = Image.fromarray(m.astype(np.uint8))
        
#         masks_list = morph_proc_masks(masks_list, kernel_size=3, iterations=2)
#         save_images_as_grid(masks_list, (len(masks_list), 1), 'results/masks_list.jpg')

#         for i, mask in enumerate(masks_list):
#             out.append(mask.resize((512,512)))
#             mask = self.process_diffedit_mask(mask, threshold=0.35, **kwargs) # binarize
            
#             # m = Image.fromarray(m.astype(np.uint8))
#             mask = mask.resize((512,512))

#             # save masks to files
#             os.makedirs('masks', exist_ok=True)
#             mask.save(f"masks/mask_{i}.jpg")
#             binary_masks_list.append(mask)

    
#         blended_img = im
#         for prompt, mask in zip(prompts, binary_masks_list):
#             ref_prompt, query_prompt = prompt
#             out.append(mask)
#             if generate_output:
#                 out.append(self.get_blended_mask(blended_img, mask))
#                 blended_img = self.inpaint(prompt=[query_prompt], image=blended_img, mask_image=mask,
#                     generator=torch.Generator(self.device).manual_seed(kwargs['seed'])).images[0]
#                 out.append(blended_img)
            
#         show_images(out)
#         save_images_as_grid(out, (len(out), 1), 'results/output_grid_image.jpg')


class MultiMaskDiffEdit(DiffEdit):
    def __init__(self, device):
        super().__init__(device)


    def multiple_masks_diffedit(self, im_path, prompts, generate_output, **kwargs):
        im_path = Path(im_path)
        out = []
        masks_list = []
        diff_masks_list = []

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
        for i in range(len(masks_list)):
            for j in range(i+1, len(masks_list)):
                diff = np.abs(np.array(masks_list[i]) - np.array(masks_list[j]))
                diff_masks_list.append(Image.fromarray(diff.astype(np.uint8)))
        
        save_images_as_grid(diff_masks_list, (len(diff_masks_list), 1), 'results/before_morph_masks_list.jpg')
        diff_masks_list = morph_proc_masks(diff_masks_list, kernel_size=3, iterations=2)
        save_images_as_grid(diff_masks_list, (len(diff_masks_list), 1), 'results/after_morph_masks_list.jpg')

        for i, mask in enumerate(masks_list):
            out.append(mask.resize((512,512)))
            mask = self.process_diffedit_mask(mask, threshold=0.35, **kwargs) # binarize
            
            # m = Image.fromarray(m.astype(np.uint8))
            mask = mask.resize((512,512))

            # save masks to files
            os.makedirs('masks', exist_ok=True)
            mask.save(f"masks/mask_{i}.jpg")
            diff_masks_list.append(mask)

    
        blended_img = im
        for prompt, mask in zip(prompts, diff_masks_list):
            ref_prompt, query_prompt = prompt
            out.append(mask)
            if generate_output:
                out.append(self.get_blended_mask(blended_img, mask))
                blended_img = self.inpaint(prompt=[query_prompt], image=blended_img, mask_image=mask,
                    generator=torch.Generator(self.device).manual_seed(kwargs['seed'])).images[0]
                out.append(blended_img)
            
        show_images(out)
        save_images_as_grid(out, (len(out), 1), 'results/output_grid_image.jpg')