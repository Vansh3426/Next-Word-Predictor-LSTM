import torch
from model import Next_word_predictor,vocab_size ,vocab ,sentece_tokenize,tokenize,max_size_of_input_sequence,padding,loss_fn


model = Next_word_predictor(vocab_size)
model.load_state_dict(torch.load('LSTM/next_word_prediction_model.pth'))
model.eval()



def text_to_index_new(sentence):
    
    tokenize_sentence_to_index =[]
    for token in sentence:
        if token  in vocab:
            token = vocab[token]
            tokenize_sentence_to_index.append(token)

    return tokenize_sentence_to_index

def padding(max_length,sequence):
    
    padded_sequences = []
    max_size = max_length
    padded_sequences.append([0]*((max_size)-len(sequence)) + sequence)
    
    return padded_sequences

def prediction(text ,model):
    
    tokenized_text =tokenize(text)
    # print(tokenized_text)
    
    
    sent_index =text_to_index_new(tokenized_text)
    # print(sent_index)
    
    
    padded_sent =padding(max_size_of_input_sequence ,sent_index)
    
    padded_sent =torch.tensor(padded_sent ,dtype=torch.long)
    # print(padded_sent.shape)
    
    logits = model(padded_sent)
    
    # print(logits.shape)
    output =torch.softmax(logits ,dim=1)
    value ,index = torch.max(output,dim=1)
    
    return text + ' ' +list(vocab.keys())[index]
    
    
    
text =" i was saying "

for i in range(10):
    output =prediction(text,model)
    
    print(output)
    
    text = output