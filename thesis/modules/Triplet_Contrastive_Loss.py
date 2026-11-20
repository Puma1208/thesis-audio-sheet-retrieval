import torch 
from torch import nn
import torch.nn.functional as F
from random import sample
import time


class Triplet_Contrastive_Loss(nn.Module):
    def __init__(self, margin=.05, K_negative=100):
        '''
        margin  :   parameter that penalizes matching snippets that were poorly embedded
        K       :   amount of non-matching passages, should be smaller that BATCH_SIZE -1
        '''
        super().__init__()
        self.margin = margin
        self.K_negative = K_negative

    def forward(self, batch_snippet_vectors, batch_excerpt_vectors, snippet_first=True, average_loss=True):
        timer_start = time.perf_counter()
        
        cosine_distances = 1-F.cosine_similarity(batch_snippet_vectors[:, None, :], batch_excerpt_vectors[None, :, :], dim=-1)
        matching_distances = torch.diagonal(cosine_distances)
        
        losses = []
        if self.K_negative>=cosine_distances.shape[1] or self.K_negative<1:
            self.K_negative = cosine_distances.shape[1]-1
        for i in range(cosine_distances.shape[0]):

            current_matching_distance = matching_distances[i]
            current_loss = 0

            valid_indeces_negative_pairs = list(range(cosine_distances.shape[0]))
            valid_indeces_negative_pairs.remove(i)
            K_negative_indeces = sample(valid_indeces_negative_pairs, self.K_negative)

            for k in K_negative_indeces:

                current_loss += max(0, self.margin + current_matching_distance.item() - cosine_distances[i][k].item())


            losses.append(torch.tensor(current_loss))
        # print(f'Losses {losses}')
        # print(torch.stack(losses))
        loss_torch = torch.mean(torch.stack(losses).float()) if average_loss else torch.sum(torch.stack(losses))
        
        # print(f'loss_torch     {loss_torch}')
        loss_torch.requires_grad = True
        # print(f'loss torch requires_grad: {loss_torch.requires_grad}')
        timer_end = time.perf_counter()
        # print(f'TRIPLET LOSS time taken: {timer_end-timer_start:.4f} seconds')
        return loss_torch

# triplet_loss = Triplet_Contrastive_Loss(margin=.1, K_negative=10)