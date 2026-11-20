from torch import nn


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

# score_passage_cnn = SheetPassageCNN(input_channels=1,
#                                             output_shape=32)
# spectrogram_passage_cnn = SpectrogramPassageCNN(input_channels=1,
#                                                         output_shape=32)