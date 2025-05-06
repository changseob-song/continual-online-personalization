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
        self.linear = nn.Linear(self.num_channels[-1] * self.window_size, self.output_size)
        
        print("\nTCN parameter #: ", sum(p.numel() for p in self.tcn.parameters()))
        print("\nFCNN parameter #: ",sum(p.numel() for p in self.linear.parameters()))
        
        # Print model summary with auto-calculated sequence length
        # summary(self, input_size=(self.input_size, self.window_size))

    def forward(self, x):
        # x shape: (batch_size, input_size, time window size = sequence length)
        y = self.tcn(x)
        # Flatten the output from the TCN layer
        y = y.flatten(start_dim=1) # Shape: (batch_size, num_channels[-1] * sequence_length)
        y = self.linear(y)
        return y
    
class LSTMModel(nn.Module):
    def __init__(self, hyperparameter_config):
        super(LSTMModel, self).__init__()
        self.input_size = hyperparameter_config['input_size']
        self.hidden_dim = hyperparameter_config['lstm_hidden_dim'] # Use a specific LSTM hidden dim
        self.num_layers = hyperparameter_config['lstm_num_layers'] # Use a specific LSTM num layers
        self.output_size = hyperparameter_config['output_size']
        self.dropout = hyperparameter_config.get('dropout', 0.2) # Use dropout from config or default

        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers,
                            batch_first=True, dropout=self.dropout if self.num_layers > 1 else 0)
        self.linear = nn.Linear(self.hidden_dim, self.output_size)
        self.init_weights()
        print("LSTM parameter #: ", sum(p.numel() for p in self.lstm.parameters()) + sum(p.numel() for p in self.linear.parameters()))

    def init_weights(self):
        # Initialize LSTM weights and biases
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                # Initialize weight matrices orthogonally
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Initialize biases to zero
                nn.init.constant_(param.data, 0)
                # Optional: Initialize forget gate bias to 1 (helps initial learning)
                # LSTM biases are ordered: [input_gate, forget_gate, cell_gate, output_gate]
                # Find the forget gate bias part (second quarter of the bias vector)
                n = param.size(0)
                start, end = n // 4, n // 2
                nn.init.constant_(param.data[start:end], 1.)

        # Initialize linear layer weights
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        # x shape: (batch_size, input_size, seq_len)
        lstm_out, (hn, cn) = self.lstm(x.transpose(1,2))
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        # hn shape: (num_layers, batch_size, hidden_dim)

        # Use the hidden state of the last layer from the last time step
        last_hidden_state = hn[-1] # Shape: (batch_size, hidden_dim)
        z = self.linear(last_hidden_state) # Shape: (batch_size, output_size)
        return z