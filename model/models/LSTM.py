import torch
import torch.nn as nn

# Define the LSTM-based model
class Model(nn.Module):
    def __init__(self, configs, dropout=0.0):
        super(Model, self).__init__()
        
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.input_size = configs.n_features

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,  # Use batch_first=True for (batch, seq, feature)
                            dropout=dropout if self.num_layers > 1 else 0.0)

        # Define the output layer
        self.fc = nn.Linear(self.hidden_size, 1, bias=True)

    def forward(self, x, *_):
        # x = torch.cat([x, batch_general], dim = -1)
        # x shape: (batch, seq_len, input_size)

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # (num_layers, batch, hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # (num_layers, batch, hidden_size)

        # Forward propagate through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden_size)

        # Use the last hidden state for output
        out = out[:, -1, :]  # (batch, hidden_size)

        # Pass through the fully connected layer
        out = self.fc(out).squeeze(1)  # (batch, output_size)
        
        return out