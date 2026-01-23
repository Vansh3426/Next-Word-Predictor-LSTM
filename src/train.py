import torch
from torch import nn
from dataset import dataloader , vocab_size
from LSTM.old_folder_structure.model import Next_word_predictor

device = 'cuda'  if torch.cuda.is_available else 'cpu'

torch.cuda.manual_seed(42)
torch.manual_seed(42)
PAD_IDX = 0


model = Next_word_predictor(vocab_size).to(device)




loss_fn =nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(params=model.parameters(),lr =0.0005)




if __name__ == "__main__":
    
    epochs = 45

    for epoch in range(epochs):
    
        total_loss = 0
        num_batch =0
        for X,y in dataloader:
        
            X ,y =X.to(device) , y.to(device)
            # print(X.shape ,y.shape)
            optimizer.zero_grad()
        
            pred =model(X)
            num_batch += 1
        
            # print(pred.shape)
            loss =loss_fn(pred ,y)
            total_loss += loss
        
            loss.backward()
        
            optimizer.step()        
        
        print(f"Epoch no : {epoch}  | total loss per epoch : {(total_loss)/(num_batch)} ")
        
        torch.save(model.state_dict() ,'LSTM/next_word_prediction_model.pth')
        
