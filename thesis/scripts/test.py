import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import torchvision.io as io
import os

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json
import argparse

# MSMD Dataset
class MSMD_Dataset(Dataset):
    def __init__(self, root_dir, pieces, snippets_subfolder='sheet_snippets', excerpts_subfolder='performances', snippets_batch_size=10, n_performances=5):
        self.dir = root_dir
        self.snippets_subfolder = snippets_subfolder
        self.excerptss_subfolder = excerpts_subfolder
        self.snippets_batch_size = snippets_batch_size
        self.pieces_folders = pieces

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

        self.snippets_excerpts_df = pd.DataFrame(columns=['snippet_folder', 'excerpt_folder', 'snippet_path', 'excerpt_path'])
        for piece_folder in tqdm(self.pieces_folders):
            piece_path = os.path.join(self.dir, piece_folder)
            snippets_dir = os.path.join(piece_path, self.snippets_subfolder)
            excerpts_dir = os.path.join(piece_path, self.excerptss_subfolder)
            print(snippets_dir)
            for performance in sorted(os.listdir(snippets_dir)[:min(len(os.listdir(snippets_dir)), n_performances)]):
            # for performance in os.listdir(snippets_dir)[:2]:
                print(f'    {performance}')
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
                    self.snippets_excerpts_df = self.snippets_excerpts_df._append({'snippet_folder':snippets_perf_dir, 
                                             'excerpt_folder':excerpts_perf_dir, 
                                             'snippet_path':snippet_name, 
                                             'excerpt_path':excerpt_name}, ignore_index=True)

# Network
class SheetPassageCNN(nn.Module):
  def __init__(self,
               input_channels:int,
               output_shape:int
              #  width:int=92,
              #  height:int=20
               ):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_channels,
                  out_channels=24,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(24)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=24,
                  out_channels=48,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(48)
    )
    self.conv_block_3 = nn.Sequential(
        nn.Conv2d(in_channels=48,
                  out_channels=96,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(96)
    )
    self.conv_block_4 = nn.Sequential(
        nn.Conv2d(in_channels=96,
                  out_channels=96,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(96)
    )
    self.conv_block_5 = nn.Sequential(
        nn.Conv2d(in_channels=96,
                  out_channels=32,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  ),
        nn.ELU(),
        nn.BatchNorm2d(32)
    )
    self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 10 * 11,
                      out_features=output_shape),
            nn.ELU(),

    )



  def forward(self, x):
    x = self.conv_block_1(x)
    # print(f'layer 1: {x.shape}')
    x = self.conv_block_2(x)
    # print(f'layer 2: {x.shape}')
    x = self.conv_block_3(x)
    # print(f'layer 3: {x.shape}')
    x = self.conv_block_4(x)
    # print(f'layer 4: {x.shape}')
    x = self.conv_block_5(x)
    # print(f'layer 5: {x.shape}')
    x = self.fully_connected(x)
    # print(f'fully connect: {x.shape}')
    return x
class SpectrogramPassageCNN(nn.Module):
  
  '''
  Might use the same as SheetPassageCNN but depends if can take different output shape
  '''
  def __init__(self,
               input_channels:int,
            #    hidden_units:int,
               output_shape:int,
            #    width:int=92,
            #    height:int=20
               ):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_channels,
                  out_channels=24,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(24)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=24,
                  out_channels=48,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(48)
    )

    self.conv_block_3 = nn.Sequential(
        nn.Conv2d(in_channels=48,
                  out_channels=96,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(96)
    )
    self.conv_block_4 = nn.Sequential(
        nn.Conv2d(in_channels=96,
                  out_channels=96,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(96)
    )
    self.conv_block_5 = nn.Sequential(
        nn.Conv2d(in_channels=96,
                  out_channels=32,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  ),
        nn.ELU(),
        nn.BatchNorm2d(32)
    )

    # self.fully_connected = nn.Linear(32, 32)
    self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 5 * 1,
                      out_features=output_shape),
            nn.ELU()
    )


  def forward(self, x):
    x = self.conv_block_1(x)
    # print(f'RNN layer 1: {x.shape}')
    x = self.conv_block_2(x)
    # print(f'RNN layer 2: {x.shape}')
    x = self.conv_block_3(x)
    # print(f'RNN layer 3: {x.shape}')
    x = self.conv_block_4(x)
    # print(f'RNN layer 4: {x.shape}')
    x = self.conv_block_5(x)
    # print(f'RNN layer 5: {x.shape}')
    x = self.fully_connected(x)
    # print(f'fully connect: {x.shape}')
    return x
class GRU_NN(nn.Module):


    '''https://blog.floydhub.com/gru-with-pytorch/'''
    def __init__(self, 
                 input_channels:int,
                 hidden_units:int, 
                 output_shape:int,
                 n_layers:int,
                 drop_prob=.2):
        
        super().__init__()
        self.hidden_units=hidden_units
        self.n_layers=n_layers
        self.output_shape=output_shape
        self.gru = nn.GRU(input_size=input_channels, 
                          hidden_size=hidden_units,
                          num_layers=n_layers,
                          bias=False,
                          batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_units, output_shape)
        self.relu = nn.ReLU()


    def forward(self, x, h=None):
        if h is None:
            h = self.init_hidden(x.shape[0])
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_units).zero_()
        return hidden
class CNN_GRU(nn.Module):
    def __init__(self, 
                 input_channels_score:int, 
                 input_channels_audio:int, 
                 output_channels_cnn_score:int,
                 output_channels_cnn_audio:int,
                 hidden_units_score:int,
                 hidden_units_audio:int,
                 output_channels:int, 
                 ):
        super().__init__()
        self.score_passage_cnn = SheetPassageCNN(input_channels=input_channels_score,
                                                 output_shape=output_channels_cnn_score)
        self.spectrogram_passage_cnn = SpectrogramPassageCNN(input_channels=input_channels_audio,
                                                             output_shape=output_channels_cnn_audio)
        self.score_passage_gru = GRU_NN(input_channels=output_channels_cnn_score,
                                        hidden_units=hidden_units_score,
                                        output_shape=output_channels,
                                        n_layers=1)
        self.spectrogram_passage_gru = GRU_NN(input_channels=output_channels_cnn_audio,
                                              hidden_units=hidden_units_audio,
                                              output_shape=output_channels,
                                              n_layers=1)
        
    def forward(self, score_snippets, audio_snippets):
        batch_size_score, seq_len_score, h_score, w_score = score_snippets.size()
        batch_size_audio, seq_len_audio, h_audio, w_audio = audio_snippets.size()

        data_score = score_snippets.view(batch_size_score*seq_len_score, 1, h_score, w_score)
        data_audio = audio_snippets.view(batch_size_audio*seq_len_audio, 1, h_audio, w_audio)

        score_cnn = self.score_passage_cnn(data_score)
        spectrogram_cnn = self.spectrogram_passage_cnn(data_audio)

        features_score = score_cnn.view(batch_size_score, seq_len_score, -1)
        features_audio = spectrogram_cnn.view(batch_size_audio, seq_len_audio, -1)


        score_cnn_gru = self.score_passage_gru(features_score)
        spectrogram_cnn_gru = self.spectrogram_passage_gru(features_audio)

        return score_cnn_gru, spectrogram_cnn_gru

    
parser = argparse.ArgumentParser(description='Embed the test set and retrieve score snippets for an audio excerpt.')
parser.add_argument('--data_dir', required=True, help='directory with the snippets created by create_dataset.py')
parser.add_argument('--model_path', required=True, help='trained model snapshot (.pt)')
parser.add_argument('--test_split', default='data/splits/msmd_test_split.json')
parser.add_argument('--test_split_snippets', default='data/splits/msmd_test_split_snippets_only.json')
args = parser.parse_args()

msmd_images_dir = args.data_dir

# model = CNN_GRU()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CNN_GRU(1, 1, 32, 32, 128, 128, 16).to(device)

model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

# Make all snippets go through the network and compute their vectors
with open(args.test_split) as f:
    testing_set_excerpt = json.load(f)
with open(args.test_split_snippets) as f:
    testing_set_snippets = json.load(f)


msmd_excerpts_test = MSMD_Dataset(root_dir=msmd_images_dir, pieces=testing_set_excerpt, n_performances=1)
msmd_excerpts_test_loader = DataLoader(msmd_excerpts_test, batch_size=1, shuffle=False)


msmd_snippets_test = MSMD_Dataset(root_dir=msmd_images_dir, pieces=testing_set_snippets, n_performances=1)
msmd_snippets_test_loader = DataLoader(msmd_snippets_test, batch_size=64, shuffle=False)

# print(len(msmd_excerpts_test.snippets_excerpts_df), len(msmd_snippets_test.snippets_excerpts_df))


############################ AUDIO EXCERPTS going through the network to compute the  vectors ##########################
audio_cnn = model.spectrogram_passage_cnn.to(device=device)
audio_rnn = model.spectrogram_passage_gru.to(device=device)
audio_vectors = []
with torch.inference_mode():
    for i, data_audio_test in tqdm(enumerate(msmd_excerpts_test_loader), total=len(msmd_excerpts_test_loader)):
        snippets_test, audio_test = data_audio_test
        batch_size_audio, seq_len_audio, h_audio, w_audio = audio_test.size()

        data_audio = audio_test.view(batch_size_audio*seq_len_audio, 1, h_audio, w_audio)

        audio_cnn_res = audio_cnn(data_audio.to(torch.float32).to(device))
        audio_rnn_res = audio_rnn(audio_cnn_res.view(batch_size_audio, seq_len_audio, -1))

        # print(audio_rnn_res[0].shape)
        audio_vectors.append(audio_rnn_res[0])
print(f'audio vectors: {len(audio_vectors)}, {(audio_vectors[0].shape)}')


############################ SCORE SNIPPETS going through the network to compute the  vectors ##########################
score_cnn = model.score_passage_cnn.to(device=device)
score_rnn = model.score_passage_gru.to(device=device)

score_vectors = []
with torch.inference_mode():

    for i, data_score_test in tqdm(enumerate(msmd_snippets_test_loader), total=len(msmd_snippets_test_loader)):
        snippets_test, audio_test = data_score_test

        batch_size_score, seq_len_score, h_score, w_score = snippets_test.size()


        data_score = snippets_test.view(batch_size_score*seq_len_score, 1, h_score, w_score)

        score_cnn_res = score_cnn(data_score.to(torch.float32).to(device))
        score_rnn_res = score_rnn(score_cnn_res.view(batch_size_score, seq_len_score, -1))

        # print(f'score: {score_rnn_res[0].shape}')
        score_vectors.append(score_rnn_res[0])

print(f'score vectors: {len(score_vectors)}, {(score_vectors[0].shape)}')
############################ Compute the similarity between the vectors and show the corresponding sequence ##########################

audio_vector_index = 30
audio_vector = audio_vectors[audio_vector_index]
best_matching_index = 0
loaders_matching = []
start_idx = best_matching_index * msmd_snippets_test_loader.batch_size
end_idx = min(start_idx + msmd_snippets_test_loader.batch_size, len(msmd_snippets_test_loader))
original_indices = list(range(start_idx, end_idx))
batches_cos_max = []
batches_indeces_max = []
for batch in score_vectors:
    print(f'batc: {batch.shape}')
    cos_similarity = F.cosine_similarity(audio_vector, batch, dim=1)
    # print(cos_similarity)

    batches_cos_max.append(torch.max(cos_similarity).item())
    batches_indeces_max.append(torch.argmax(cos_similarity).item())

print(batches_cos_max)
print(batches_indeces_max)
############################ Visualise excerpts ##########################

# data = msmd_excerpts_test[audio_vector_index]
# print(data[1].shape)


# # for i in range(starting_snippet_excerpt_index, starting_snippet_excerpt_index+self.snippets_batch_size):
# #             snippet = io.read_image(os.path.join(snippets_dir, all_snippets_perf[i]), mode=io.ImageReadMode.GRAY)
# #             snippets.append(snippet)
        
# plt.figure('Alignment', figsize=(20, 10))
# # i = 1
# for i, im in enumerate(data[1]):
#     plt.subplot(1, data[1].shape[0], i+1)

#     plt.imshow(im, cmap='magma', origin='lower')
# # plt.ylim(note_object.middle[0]+100, note_object.middle[0]-100)
# # plt.subplot(2, 1, 2)
# # plt.imshow(test_spectrogram)
# # plt.xlim(onset_frame-100, onset_frame+100)
# # plt.show()
# plt.savefig('alignment.png', bbox_inches='tight')