# DiffEdit

This project proposes an improvement of a model suggested by the [DiffEdit paper](https://arxiv.org/pdf/2210.11427.pdf).

This is a class project as part of CS236610 - Diffusion Models course @ Technion.  

<p align="center">
    <a href="https://www.linkedin.com/in/nitzan-bar-9ab896146/">Nitzan Bar</a>  •  
    <a href="https://www.linkedin.com/in/tuval-gelvan-ab87b4136/">Tuval Gelvan</a>
</p>


## Files in the repository
|File name         | Purpsoe |
|----------------------|------|
|`code/main.py`| main code file|
|`code/multi_mask_diff_edit.py`| suggested implementation code|
|`code/Multi_Diffusion.ipynb`|  Multi-Diffusion code - Jupiter Notebook format|
|`data`|  input images to the model|
|`results`|  resulted images|
|`requirements.txt`|  python packages dependencies|


## Introduction
DiffEdit is a zero-shot inpainting method given a text-to-image denoising diffusion model, requiring no additional fine-tuning or parameters.

The main innovations pioneered in DiffEdit are:

Automated mask inference given the original and new conditioning text using the spatial distribution of the difference in conditioned noise estimates, highlighting the locations that are predicted to change the most between conditioning on the original and new texts.
Using inverse sampling from model-estimated noise to provide a noised version of the unmasked portion of the input image for each sampling step, as opposed to mixing the unmasked portion with random IID Gaussian noise as in SDEdit and RePaint.

## DiffEdit Limitations:​
- Does not support multiple prompts, it can only process a single prompt at a time.​
- Fixed threshold value is being used for converting the generated mask into a binary mask.​
- Can not handle properly overlapping areas in the image, as seen in the following example.​
​
## Suggested Improvements
- Improve step 1 by generating multiple masks {M_i }_(i=0)^N given N prompts.
- Generate DiffEdit mask for each prompt separately.
- Instead of using a fixed value threshold for all masks, try to use a threshold that computed for each mask separately. (e.g. - taking its median value)
- Generate new mask for overlapping areas in the image.
- Try to improve results by using [Multi-Diffusion paper](https://arxiv.org/abs/2302.08113) approach.

## Results
![alt text](https://github.com/Technion-Nitzan-Bar-Tuval-Gelvan/DiffEdit-Project/blob/main/figures/results1.png)
![alt text](https://github.com/Technion-Nitzan-Bar-Tuval-Gelvan/DiffEdit-Project/blob/main/figures/results2.png)


## Presentation
- [Recording](https://www.youtube.com/watch?v=zgE8nUYU-ng&t=3592s) of in-class project presentation.
- Slides can be found [here](https://github.com/Technion-Nitzan-Bar-Tuval-Gelvan/DiffEdit-Project/blob/main/diffusion_presentation_final_project.pptx)

## References
[Couairon, G., Verbeek, J., Schwenk, H. and Cord, M., 2022. Diffedit: Diffusion-based semantic image editing with mask guidance](https://arxiv.org/pdf/2210.11427.pdf)
[Bar-Tal, O., Yariv, L., Lipman, Y. and Dekel, T., 2023. Multidiffusion: Fusing diffusion paths for controlled image generation](https://arxiv.org/abs/2302.08113)
