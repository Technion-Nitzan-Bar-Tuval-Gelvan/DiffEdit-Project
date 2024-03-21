
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



class DiffEdit:
    def __init__(self, device):
        self.device = device

        # Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

        # Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

        # The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

        # The noise scheduler
        # hyper parameters match those used during training the model
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        # To the GPU we go!
        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.unet = self.unet.to(self.device)

        # vae model trained with a scale term to get closer to unit variance
        self.vae_magic = 0.18215 

        # Load RunwayML's Inpainting Model
        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained('runwayml/stable-diffusion-inpainting',
                                                                      revision="fp16",
                                                                      torch_dtype=torch.float16).to(self.device)
    
    def image2latent(self, im):
        im = tfms.ToTensor()(im).unsqueeze(0)
        with torch.no_grad():
            latent = self.vae.encode(im.to(self.device)*2-1);
        latent = latent.latent_dist.sample() * self.vae_magic
        return latent

    def latents2images(self, latents):
        latents = latents/self.vae_magic
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0,1)
        imgs = imgs.detach().cpu().permute(0,2,3,1).numpy()
        imgs = (imgs * 255).round().astype("uint8")
        imgs = [Image.fromarray(i) for i in imgs]
        return imgs

    def get_embedding_for_prompt(self, prompt):
        max_length = self.tokenizer.model_max_length
        tokens = self.tokenizer([prompt],padding="max_length",max_length=max_length,truncation=True,return_tensors="pt")
        with torch.no_grad():
            embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
        return embeddings
    
    # Given a starting image latent and a prompt; predict the noise that should be removed to transform
# the noised source image to a denoised image guided by the prompt.
    def predict_noise(self, text_embeddings,im_latents,seed=torch.seed(),guidance_scale=7,strength=0.5,**kwargs):
        num_inference_steps = 50            # Number of denoising steps

        generator = torch.manual_seed(seed)   # Seed generator to create the inital latent noise

        uncond = self.get_embedding_for_prompt('')
        text_embeddings = torch.cat([uncond, text_embeddings])

        # Prep Scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * 1 * 1, device=self.device)

        start_step = init_timestep
        noise = torch.randn_like(im_latents)
        latents = self.scheduler.add_noise(im_latents,noise,timesteps=timesteps)
        latents = latents.to(self.device).float()

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:].to(self.device)

        noisy_latent = latents.clone()

        noise_pred = None
        for i, tm in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, tm)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, tm, encoder_hidden_states=text_embeddings)["sample"]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            u = noise_pred_uncond
            g = guidance_scale
            t = noise_pred_text

            # perform guidance
            noise_pred = u + g * (t - u)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, tm, latents).prev_sample

        return self.latents2images(latents)[0],noise_pred
    
    def calc_diffedit_samples(self, encoded,prompt1,prompt2,n=10,**kwargs):
        diffs=[]
        # So we can reproduce mask generation we generate a list of n seeds
        torch.manual_seed(torch.seed() if 'seed' not in kwargs else kwargs['seed'])
        seeds = torch.randint(0,2**62,(10,)).tolist()
        for i in range(n):
            kwargs['seed'] = seeds[i] # Important to use same seed for the two noise samples
            emb1 = self.get_embedding_for_prompt(prompt1)
            _im1,n1 = self.predict_noise(emb1,encoded,**kwargs)
            emb2 = self.get_embedding_for_prompt(prompt2)
            _im2,n2 = self.predict_noise(emb2,encoded,**kwargs)

            # Aggregate the channel components by taking the euclidean distance.
            diffs.append((n1-n2)[0].pow(2).sum(dim=0).pow(0.5)[None])
        all_masks = torch.cat(diffs)
        return all_masks

    # Given an image latent and two prompts; generate a grayscale diff by sampling the noise predictions
    # between the prompts.
    def calc_diffedit_diff(self, im_latent,p1,p2,**kwargs):
        m = self.calc_diffedit_samples(im_latent,p1,p2,**kwargs)
        m = m.mean(axis=0) # average samples together
        m = (m-m.min())/(m.max()-m.min()) # rescale to interval [0,1]
        m = (m*255.).cpu().numpy().astype(np.uint8)
        m = Image.fromarray(m)
        return m

    # Try to improve the mask thru convolutions etc
    # assume m is a PIL object containing a grayscale 'diff'
    def process_diffedit_mask(self, m,threshold=0.35,**kwargs):
        m = np.array(m).astype(np.float32)
        m = cv2.GaussianBlur(m,(5,5),1)
        m = (m>(255.*threshold)).astype(np.float32)*255
        m = Image.fromarray(m.astype(np.uint8))
        return m

    # Given an image latent and two prompts; generate a binarized mask (PIL) appropriate for inpainting
    def calc_diffedit_mask(self, im_latent,p1,p2,**kwargs):
        m = self.calc_diffedit_diff(im_latent,p1,p2,**kwargs)
        m = self.process_diffedit_mask(m,**kwargs)
        m = m.resize((512,512))
        return m
    
    # Composite the mask over the provided image; for demonstration purposes
    def get_blended_mask(self, im, mask_gray): # Both expected to be PIL images
        mask_rgb = mask_gray.convert('RGB')
        return Image.blend(im,mask_rgb,0.40)

    # Show the original image, the original image with mask and the resulting inpainted image
    def demo_diffedit(self, im_path,p1,p2,**kwargs):
        im_path = Path(im_path)
        out = []

        im = Image.open(im_path).resize((512,512))
        im_latent = self.image2latent(im)
        out.append(im)

        if 'seed' not in kwargs: kwargs['seed'] = torch.seed()
        mask = self.calc_diffedit_mask(im_latent,p1,p2,**kwargs)
        out.append(self.get_blended_mask(im,mask))
        out.append(self.inpaint(prompt=[p2],image=im,mask_image=mask,
            generator=torch.Generator(self.device).manual_seed(kwargs['seed'])).images[0])
        show_images(out)