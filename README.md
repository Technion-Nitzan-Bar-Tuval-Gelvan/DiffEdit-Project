# DiffEdit

This project proposes an improvement of a model suggested by the [DiffEdit paper](https://arxiv.org/pdf/2210.11427.pdf).

This is a class project as part of CS236610 - Diffusion Models course @ Technion.  

<p align="center">
    <a href="https://www.linkedin.com/in/nitzan-bar-9ab896146/">Nitzan Bar</a>  •  
    <a href="(https://www.linkedin.com/in/tuval-gelvan-ab87b4136/)">Tuval Gelvan</a>
</p>


## Files in the repository
|File name         | Purpsoe |
|----------------------|------|
|`code/main.py`| main code file|
|`code/multi_mask_diff_edit.py`| suggested implementation code|
|`code/Multi_Diffusion.ipynb`|  Multi-Diffusion code - Jupiter Notebook format|
|`data`|  input images to the model|
|`results`|  resulted image|


## Introduction
DiffEdit is a zero-shot inpainting method given a text-to-image denoising diffusion model, requiring no additional fine-tuning or parameters.

The main innovations pioneered in DiffEdit are:

Automated mask inference given the original and new conditioning text using the spatial distribution of the difference in conditioned noise estimates, highlighting the locations that are predicted to change the most between conditioning on the original and new texts.
Using inverse sampling from model-estimated noise to provide a noised version of the unmasked portion of the input image for each sampling step, as opposed to mixing the unmasked portion with random IID Gaussian noise as in SDEdit and RePaint.

## Suggested Solution

## Results

![alt text](https://github.com/NitzanShitrit/EEGClassification/blob/main/images/graphs.PNG)
![alt text](https://github.com/NitzanShitrit/EEGClassification/blob/main/images/table.PNG)


## Presentation
- [Recording](https://youtu.be/V5hxXmG1A9U) of in-class project presentation (Hebrew only)
- Slides can be found [here](https://github.com/NitzanShitrit/EEGClassification/blob/main/slides.pptx)

## References
