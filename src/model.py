import torch
from torch import nn
from dataset import PAD_IDX

class Next_word_predictor(nn.Module):
    def __init__(self ,vocab_size):
        super().__init__()
        
        self.embedding =nn.Embedding(num_embeddings=vocab_size,embedding_dim=100 , padding_idx=PAD_IDX)
        self.lstm =nn.LSTM(input_size=100 ,hidden_size=128 ,num_layers=2, batch_first=True)
        self.linear =nn.Linear(128 , vocab_size)
    
    def forward(self,x):
        embeddings =self.embedding(x)
        each_timestamp_hidden_state_output ,(final_hidden_state , final_cell_state) = self.lstm(embeddings)
        final_output =self.linear(final_hidden_state[-1])
        return final_output