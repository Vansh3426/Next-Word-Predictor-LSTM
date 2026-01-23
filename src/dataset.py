import torch
from torch.utils.data import Dataset ,DataLoader
from preprocessing import tokenize , dictionary , build_sentences ,text_to_index ,sentece_tokenize ,build_training_data, size,padding 


device = 'cuda'  if torch.cuda.is_available else 'cpu'

torch.cuda.manual_seed(42)
torch.manual_seed(42)

with open("LSTM/LSTM-word-prediction/1661-0.txt", "r", encoding="utf-8") as f:
    text = f.read()
   
   
tokens =tokenize(text)

vocab =dictionary(tokens)
vocab_size =len(vocab)
print(vocab_size)



sentences =build_sentences(text) ### input sentences

tokenize_sentence =sentece_tokenize(sentences)

input_sentences =text_to_index(tokenize_sentence ,vocab)


training_sequences , labelled_sequences =build_training_data(input_sentences)

max_size_of_input_sequence =size(training_sequences)

max_length = 65


padded_training_sequences = padding(max_length,training_sequences) ## this goes in the embedding layer 
PAD_IDX = 0



padded_training_sequences =torch.tensor(padded_training_sequences, dtype=torch.long)
labelled_sequences =torch.tensor(labelled_sequences, dtype=torch.long)

# print(padded_training_sequences.shape , labelled_sequences.shape)


# with open("vocab.json", "w") as f:
#     json.dump(vocab, f)





class MyDataset(Dataset):
    
    def __init__(self ,x,y):
        super().__init__()
        self.X = x
        self.y =y
        
    def __len__(self):
       
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
my_dataset = MyDataset(padded_training_sequences,labelled_sequences)

dataloader =DataLoader(dataset= my_dataset,batch_size=32 ,shuffle=True)