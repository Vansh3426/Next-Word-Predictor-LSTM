import torch 
from torch import nn
from torch.utils.data import DataLoader,Dataset
import spacy
import json


device = 'cuda'  if torch.cuda.is_available else 'cpu'

torch.cuda.manual_seed(42)
torch.manual_seed(42)

nlp = spacy.load("en_core_web_md")

with open("LSTM/LSTM-word-prediction/1661-0.txt", "r", encoding="utf-8") as f:
    text = f.read()
    

def tokenize(text):
    doc =nlp(text)
    
    tokenized_text =[]
    for token in doc:
        
        
        # token = token.split(" ")
        if not token.is_punct and not token.is_quote and not token.is_space:
            tokenized_text.append(token.lower_)
    return tokenized_text

tokens =tokenize(text)



def dictionary(list):
    
    token_list =list
    vocab ={"<pad>":0,"<unk>":1}
    
    for word in token_list:
        if word not in vocab:
            vocab[word] = len(vocab)
        
        
    return vocab
    
vocab =dictionary(tokens)
vocab_size =len(vocab)
print(vocab_size)



with open("vocab.json", "w") as f:
    json.dump(vocab, f)



def build_sentences(text):
    doc =nlp(text)
    
    input_sentences_list =[]
    
    for sentence in doc.sents:
        
     input_sentences_list.append(sentence.text.replace('\n',' ').strip())
    
    return input_sentences_list 
 
    

sentences =build_sentences(text) ### input sentences

def sentece_tokenize(sentence_list):
    new_sentences =[]
    for sentence in sentence_list:
        sent =tokenize(sentence)
        new_sentences.append(sent)
    
    return new_sentences



tokenize_sentence =sentece_tokenize(sentences)


def text_to_index(list_sentences):
    
    tokenize_sentence_to_index =[]
    for sentence in list_sentences:
        sentence_to_index =[]
        tokenize_sentence_to_index.append(sentence_to_index)
        for token in sentence:
            token = vocab[token]
            sentence_to_index.append(token)

    return tokenize_sentence_to_index


input_sentences =text_to_index(tokenize_sentence)

# print(input_sentences)



# Creating the training data for our lstm model , by language modeling technique 

def build_training_data(text):
    
    training_data =[]
    labelled_data =[]
    
    for sentence in text:
        
        for i in range(1,len(sentence)):
           
            train_sent =sentence[:i]
          
            label_sent=sentence[i]
            training_data.append(train_sent)
            labelled_data.append(label_sent)
    
    
    return training_data , labelled_data

training_sequences , labelled_sequences =build_training_data(input_sentences)


def size(text):
    size_list = []
    for row in text:
    
        size_list.append(len(row))
    return max(size_list)

max_size_of_input_sequence =size(training_sequences)

max_length = 65

def padding(max_length,train_sequences):
    
    padded_sequences = []
    max_size = max_length
    
    for sequence in train_sequences:
        sequence = sequence[-max_size:]
        padded_sequences.append([0]*((max_size)-len(sequence)) + sequence)
    
    return padded_sequences

padded_training_sequences = padding(max_length,training_sequences) ## this goes in the embedding layer 
PAD_IDX = 0



padded_training_sequences =torch.tensor(padded_training_sequences, dtype=torch.long)
labelled_sequences =torch.tensor(labelled_sequences, dtype=torch.long)

# print(padded_training_sequences.shape , labelled_sequences.shape)


#### Creatig datasets and dataloader 


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

# x ,y = next(iter(dataloader))
# torch.set_printoptions(profile="full")
# print(x)
# torch.set_printoptions(profile="default") 


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
        
model = Next_word_predictor(vocab_size).to(device)



loss_fn =nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(params=model.parameters(),lr =0.0005)


# x,y =next(iter(dataloader))
# X =x[0].unsqueeze(dim=0)
# print(X.shape)
# a = nn.Embedding(num_embeddings=vocab_size,embedding_dim=100)
# b  =nn.LSTM(100 ,150,batch_first=True) 
# c  =nn.Linear(150,vocab_size)
# E = a(X)
# print(E.shape)
# output1 , output2 =b(E) 
# hd ,cs =output2
# print(hd.shape  ,  cs.shape)
# final =c(output1)
# print(final.shape)

## training
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
        
