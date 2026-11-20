from torch import nn

class GRU_NN(nn.Module):


    '''https://blog.floydhub.com/gru-with-pytorch/'''
    def __init__(self, 
                 input_channels:int,
                 hidden_units:int, 
                 output_shape:int,
                 n_layers:int
                 ):
        
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

# score_passage_gru = GRU_NN(input_channels=1,
#                                         hidden_units=1,
#                                         output_shape=32,
#                                         n_layers=1)
# spectrogram_passage_gru = GRU_NN(input_channels=1,
#                                               hidden_units=1,
#                                               output_shape=32,
#                                               n_layers=1)
        