"""
Evaluate a trained CNN-GRU: embed all the test audio excepts and score
snippets, then find the score snippets most similar to a query exerpt
(with cosine similarity).

Usage (from the repository root):
    python -m thesis.script.test --data_dir <snippets_dir> --model_path <snapshot.pt>
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json
import argparse

from thesis.modules.MSMD_Dataset import MSMD_Dataset
from thesis.modules.CNN_GRU import CNN_GRU


parser = argparse.ArgumentParser(description='Embed the test set and retrieve score snippets for an audio excerpt.')
parser.add_argument('--data_dir', required=True, help='directory with the snippets created by create_dataset.py')
parser.add_argument('--model_path', required=True, help='trained model snapshot (.pt)')
parser.add_argument('--test_split', default='data/splits/msmd_test_split.json')
parser.add_argument('--test_split_snippets', default='data/splits/msmd_test_split_snippets_only.json')
parser.add_argument('--output_dim', type=int, default=64, help="must match the trained model")
parser.add_argument('--query_index', type=int, default=30m help='index of the audio excerpt used as query')
args = parser.parse_args()

msmd_images_dir = args.data_dir

# model = CNN_GRU()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CNN_GRU(1, 1, 32, 32, 128, 128, args.output_dim).to(device)

snapshot = torch.load(args.model_path, map_location=device)
model.load_state_dict(snapshot['MODEL_STATE'])
model.eval()

# Make all snippets go through the network and compute their vectors
with open(args.test_split) as f:
    testing_set_excerpt = json.load(f)
with open(args.test_split_snippets) as f:
    testing_set_snippets = json.load(f)


msmd_excerpts_test = MSMD_Dataset(dir=msmd_images_dir, pieces=testing_set_excerpt, n_performances=1)
msmd_excerpts_test_loader = DataLoader(msmd_excerpts_test, batch_size=1, shuffle=False)


msmd_snippets_test = MSMD_Dataset(dir=msmd_images_dir, pieces=testing_set_snippets, n_performances=1)
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

audio_vector = audio_vectors[args.query_index]
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