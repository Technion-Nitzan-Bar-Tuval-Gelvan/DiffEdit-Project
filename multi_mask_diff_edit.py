
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


class MultiMaskDiffEdit(DiffEdit):
    def __init__(self):
        super().__init__()


    def multiple_masks_diffedit(self, im_path, prompts, generate_output, **kwargs):
        im_path = Path(im_path)
        out = []

        im = Image.open(im_path).resize((512,512))
        im_latent = self.image2latent(im)
        out.append(im)

        for prompt in prompts:
            ref_prompt, query_prompt = prompt
            if 'seed' not in kwargs: kwargs['seed'] = torch.seed()
            diff_mask = self.calc_diffedit_diff(im_latent,ref_prompt,query_prompt,**kwargs)
            out.append(diff_mask)
            mask = self.calc_diffedit_mask(im_latent,ref_prompt,query_prompt,**kwargs)
            out.append(mask)
            if generate_output:
                out.append(self.get_blended_mask(im,mask))
                out.append(self.inpaint(prompt=[query_prompt],image=im,mask_image=mask,
                    generator=torch.Generator(self.device).manual_seed(kwargs['seed'])).images[0])
            show_images(out)
            save_images_as_grid(out, (len(out), 1), 'output_grid_image.jpg')
