# Margarita Naryzhnyaya

import numpy as np 
import torch
import os 
from torch.utils.data import random_split
import json

def main():
    msmd_piece_images_dir = '/home/I6194030/MSMD_SNIPPETS_2'
    path_train_split = 'data'
    path_valid_split = 'data'

    pieces = sorted(os.listdir(msmd_piece_images_dir))
    generator = torch.Generator().manual_seed(45)


    train_split, validation_split = random_split(pieces, [.7, .3], generator=generator)
    train_list = [pieces[i] for i in train_split.indices]
    validation_list = [pieces[i] for i in validation_split.indices]

    
    with open(f'{path_train_split}/msmd_train_split.json', 'w') as f:
        json.dump(train_list, f, indent=4)
    with open(f'{path_valid_split}/fake_validation_split.json', 'w') as f:
        json.dump(validation_list, f, indent=4)
        


if __name__ == '__main__':
    """
    main
    """
    main()
