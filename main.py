import torch
import argparse
from datetime import datetime
from fastdownload import FastDownload
from diff_edit import DiffEdit

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)


    parser = argparse.ArgumentParser(description='Main')
    # Add arguments
    parser.add_argument('--image_path', type=str, help='Training file', default='DiffEdit-Project/data/fruitbowl_scaled.jpg')
    parser.add_argument('--cuda_device', type=int, help='number of gpu to use', default=3)
    parser.add_argument('--seed', type=int, help='seed', default=42)

    # Parse the arguments
    args = parser.parse_args()

    # set cuda device to use
    torch.cuda.set_device(int(args.cuda_device))
    date = datetime.now()

    diff_edit = DiffEdit(device=args.cuda_device)
    prompts = [('Horse','Zebra'),
               ('Grass','Sand')]

    diff_edit.multiple_masks_diffedit(im_path=args.image_path,
                                      prompts=prompts,
                                      seed=args.seed)


if __name__ == "__main__":
    main()