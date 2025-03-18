# ICORR_Header_Model.py
import torch.nn as nn
from torchsummary import summary

# Define the TCN Model with multiple convolutional layers per TemporalBlock
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, number_of_layers, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        layers = []
        in_channels = n_inputs
        for i in range(number_of_layers):
            padding = (kernel_size - 1) * dilation
            layers += [nn.ConstantPad1d((padding, 0), 0),
                       nn.Conv1d(in_channels, n_outputs, kernel_size, stride=stride, dilation=dilation),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
            in_channels = n_outputs
            
        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.network:
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
        if self.downsample is not None:
            nn.init.xavier_uniform_(self.downsample.weight)

    def forward(self, x):
        out = self.network(x)
        res = x if self.downsample is None else self.downsample(x)
        # Ensure the shapes match for addition
        out = out[:, :,  -res.size(2):]  # Trim to match the residual size
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, number_of_layers, kernel_size=2, dropout=0.2, dilations=None):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_blocks = len(num_channels)
        
        for i, dilation in zip(range(num_blocks), dilations):
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, number_of_layers, kernel_size, stride=1,
                                        dilation=dilation, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Define the TCN Network (replacing LSTM)
class TCNModel(nn.Module):
    def __init__(self, hyperparameter_config):
        super(TCNModel, self).__init__()
        
        # Configuration parameters
        self.input_size = hyperparameter_config['input_size']  # Number of sensor inputs (6 for each IMU)
        self.output_size = hyperparameter_config['output_size']  # Number of joint angles
        self.num_channels = hyperparameter_config['num_channels']  # Channels per TemporalBlock
        self.kernel_size = hyperparameter_config['kernel_size']  # Kernel size for convolutional layers
        self.number_of_layers = hyperparameter_config['number_of_layers']
        self.dropout = hyperparameter_config['dropout']  # Dropout value
        self.dilations = hyperparameter_config['dilations']  # Dilations for TemporalBlocks
        self.window_size = hyperparameter_config['window_size']  # Get the sequence length
        
        self.tcn = TemporalConvNet(self.input_size, self.num_channels, self.number_of_layers, self.kernel_size, self.dropout, self.dilations)
        self.linear = nn.Linear(self.num_channels[-1], self.output_size)
        
        print("\nTCN parameter #: ", sum(p.numel() for p in self.tcn.parameters()))
        print("\nFCNN parameter #: ",sum(p.numel() for p in self.linear.parameters()))
        
        # Print model summary with auto-calculated sequence length
        # summary(self, input_size=(self.input_size, self.window_size))

    def forward(self, x):
        # x shape: (batch_size, input_size, time window size = sequence length)
        y = self.tcn(x)
        y = y[:, :, -1]  # Take the last time step
        y = self.linear(y)
        return y