from torch import nn
from .PassageCNN import SheetPassageCNN, SpectrogramPassageCNN
from .GRU import GRU_NN

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

# cnn_gru = CNN_GRU(1, 1, 32, 32, 128, 128, 64)

