import torch
from torch import nn
import torch.nn.functional as F
from random import sample

class Triplet_Contrastive_Loss(nn.Module):
    def __init__(self, margin=.05, K_negative=100):
        '''
        margin  :   parameter that penalizes matching snippets that were poorly embedded
        K       :   amount of non-matching passages, should be smaller that BATCH_SIZE -1
        '''
        super().__init__()
        self.margin = margin
        self.K_negative = K_negative

    def forward(self, batch_snippet_vectors, batch_excerpt_vectors, average_loss=True):

        cosine_distances = 1-F.cosine_similarity(batch_snippet_vectors[:, None, :], batch_excerpt_vectors[None, :, :], dim=-1)
        matching_distances = torch.diagonal(cosine_distances)
        batch_size = cosine_distances.shape[0]
        k_negative = min(self.K_negative, batch_size-1)
        if k_negative < 1:
            return cosine_distances.sum() * 0

        losses = []
        for i in range(batch_size):
            negatives = torch.tensor(sample([j for j in range(batch_size) if j != i], k_negative))
            losses.append(torch.clamp(self.margin + matching_distances[i] - cosine_distances[i, negatives], min=0).sum())

        losses = torch.stack(losses)
        return losses.mean() if average_loss else losses.sum()