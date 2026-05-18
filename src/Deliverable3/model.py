import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Transformers don't have built-in memory of sequence order like LSTMs do.
    We must inject positional math so it knows the chronological order of the audio.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a matrix of [max_len, d_model] representing the positional math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: [1, max_len, d_model]

    def forward(self, x):
        # x shape: [Batch, Time, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class AudioTransformer(nn.Module):
    def __init__(self, num_mels=64, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        """
        Sequence-to-Sequence Audio Transformer for Noise Suppression.
        Complexity: Uses Self-Attention to map Noisy Spectrograms to Clean Spectrograms.
        """
        super(AudioTransformer, self).__init__()
        self.num_mels = num_mels
        
        # 1. Project the 64 Mels into a higher-dimensional embedding space (d_model)
        self.input_projection = nn.Linear(num_mels, d_model)
        
        # 2. Add chronologic time context
        self.positional_encoding = PositionalEncoding(d_model)
        
        # 3. The Sequence Processor (Transformer Encoder)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=dropout,
            batch_first=True # PyTorch handles [Batch, Time, Features] smoothly
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. Project back down to the 64 Mel audio features
        self.output_projection = nn.Linear(d_model, num_mels)

    def forward(self, x):
        # Input 'x' comes from DataLoader: [Batch, Channels(1), Mels(64), Time]
        
        # Step A: Reshape for Sequence Modeling
        # Squeeze out the channel dimension: [Batch, 64, Time]
        x = x.squeeze(1) 
        # Swap axes to make Time the sequence dimension: [Batch, Time, 64]
        x = x.permute(0, 2, 1) 
        
        # Step B: Pass through the Transformer blocks
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        
        # Step C: Reconstruct the Clean Audio Spectrogram
        x = self.output_projection(x)
        
        # Swap back to original shape: [Batch, 1, 64, Time]
        x = x.permute(0, 2, 1).unsqueeze(1)
        
        return x

# Quick test to ensure shapes match
if __name__ == "__main__":
    # Create dummy data: Batch=16, Channel=1, Mels=64, Time_steps=100
    dummy_noisy_sequence = torch.randn(16, 1, 64, 100)
    
    model = AudioTransformer(num_mels=64)
    output_clean_sequence = model(dummy_noisy_sequence)
    
    print(f"Input Noisy Shape:  {dummy_noisy_sequence.shape}")
    print(f"Output Clean Shape: {output_clean_sequence.shape}")