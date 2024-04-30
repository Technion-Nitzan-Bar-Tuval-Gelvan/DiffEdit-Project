import torch
import argparse
from datetime import datetime
from fastdownload import FastDownload
from diff_edit import DiffEdit
from multi_mask_diff_edit import MultiMaskDiffEdit

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)


    parser = argparse.ArgumentParser(description='Main')
    # Add arguments
    parser.add_argument('--image_path', type=str, help='Training file', default='data/giraffe.jpg')
    parser.add_argument('--cuda_device', type=int, help='number of gpu to use', default=3)
    parser.add_argument('--seed', type=int, help='seed', default=42)

    # Parse the arguments
    args = parser.parse_args()

    # set cuda device to use
    torch.cuda.set_device(int(args.cuda_device))
    date = datetime.now()

    diff_edit = MultiMaskDiffEdit(device=args.cuda_device)
    # horse_scaled.jpg - threshold 
    # prompts = [               
    #             ('Grass','Sand'),
    #            ('Horse','Zebra'),
    #            ]

    # fruitbowl_scaled.jpg - threshold 0.45
    # prompts = [('Chopping board','Frame'),
    #            ('Fruits','Rice'),
    #            ('Bowl','Basket'),
    #            ]

    # prompts = [('flower box', "basket"),
    #           ("dog", "cat"),]
    
    # prompts = [('bike', "motorcycle"),
    #           ("dog", "cat"),]

    prompts = [('giraffe', "cow"),
              ("tree", "lamp"),]

    diff_edit.multiple_masks_diffedit(im_path=args.image_path,
                                      prompts=prompts,
                                      seed=args.seed,
                                      generate_output=True)


if __name__ == "__main__":
    main()