from tqdm import tqdm 
import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from  torchvision import transforms
import torchvision.io as io


class MSMD_Dataset(Dataset):
    '''
    Loading score snippets and the corresponding audio excerpts 
    to create a dataset of aligned images from both modalities
    '''
    def __init__(self, 
                 dir:str, 
                 pieces:list, 
                 snippets_folder_name:str='sheet_snippets', 
                 excerpts_folder_name:str='performances', 
                 snippets_batch_size=10, 
                 n_performances=5):
        self.dir = dir
        self.pieces_folders = pieces
        self.snippets_folder_name = snippets_folder_name
        self.excerpts_folder_name = excerpts_folder_name
        self.snippets_batch_size = snippets_batch_size

        self.transform = transforms.ToTensor()
        self.define_indeces_snippets(n_performances=n_performances)
        self.len = len(self.snippets_excerpts_df)
    
    def __len__(self):
        if self.len is None:
            self.len = len(self.snippets_excerpts_df)
        return self.len
    
    def __getitem__(self, index):
        '''
        Take N images from the piece-performance 
        Starting from index in the dataframe of images we can start from 
        '''
        starting_row = self.snippets_excerpts_df.iloc[index]
        snippets_dir = starting_row['snippet_folder']
        excerpts_dir = starting_row['excerpt_folder']

        all_snippets_perf = sorted(os.listdir(snippets_dir), key=lambda x:int(x.split('_')[-1].split('.')[0]))
        all_excerpts_perf = sorted(os.listdir(excerpts_dir), key=lambda x:int(x.split('_')[-1].split('.')[0]))

        starting_snippet_excerpt_index = int(starting_row['snippet_path'].split('_')[-1].split('.')[0])

        snippets = []
        excerpts = []
        for i in range(starting_snippet_excerpt_index, starting_snippet_excerpt_index+self.snippets_batch_size):
            snippet = io.read_image(os.path.join(snippets_dir, all_snippets_perf[i]), mode=io.ImageReadMode.GRAY)
            snippets.append(snippet)
            excerpt = io.read_image(os.path.join(excerpts_dir, all_excerpts_perf[i]), mode=io.ImageReadMode.GRAY)
            excerpts.append(excerpt)
        
        return torch.cat(snippets, dim=0), torch.cat(excerpts, dim=0)


    def define_indeces_snippets(self, n_performances):
        '''
        Define a Data Frame that stores the indeces of snippets that can be the start of sequences 
        depending on the snippet_amount parameter
        '''

        # self.snippets_excerpts_df = pd.DataFrame(columns=['snippet_folder', 'excerpt_folder', 'snippet_path', 'excerpt_path'])
        snippets_excerpts_rows = []
        for piece_folder in tqdm(self.pieces_folders, colour='#ffebb2', desc='Loading Dataset'):
            piece_path = os.path.join(self.dir, piece_folder)
            snippets_dir = os.path.join(piece_path, self.snippets_folder_name)
            excerpts_dir = os.path.join(piece_path, self.excerpts_folder_name)
            # print(snippets_dir)
            for performance in sorted(os.listdir(snippets_dir)[:min(len(os.listdir(snippets_dir)), n_performances)]):
            # for performance in os.listdir(snippets_dir)[:2]:
                # print(performance)
                snippets_perf_dir = os.path.join(snippets_dir, performance)
                excerpts_perf_dir = os.path.join(excerpts_dir, performance)

                amount_snippets = len(os.listdir(snippets_perf_dir))
                if amount_snippets<self.snippets_batch_size:
                    continue
                max_image_index_performance = amount_snippets-self.snippets_batch_size
                
                # Iterate over the images to store them in the dataframe
                all_snippets_perf = sorted(os.listdir(excerpts_perf_dir), key=lambda x:int(x.split('_')[-1].split('.')[0]))
                all_excerpts_perf = sorted(os.listdir(snippets_perf_dir), key=lambda x:int(x.split('_')[-1].split('.')[0]))
                for snippet_name, excerpt_name in zip(all_snippets_perf[:max_image_index_performance+1], 
                                            all_excerpts_perf[:max_image_index_performance+1]):
                    # self.snippets_excerpts_df = self.snippets_excerpts_df._append({'snippet_folder':snippets_perf_dir, 
                    #                          'excerpt_folder':excerpts_perf_dir, 
                    #                          'snippet_path':snippet_name, 
                    #                          'excerpt_path':excerpt_name}, ignore_index=True)
                    snippets_excerpts_rows.append({'snippet_folder':snippets_perf_dir, 
                                             'excerpt_folder':excerpts_perf_dir, 
                                             'snippet_path':snippet_name, 
                                             'excerpt_path':excerpt_name})
        self.snippets_excerpts_df = pd.DataFrame(snippets_excerpts_rows)

