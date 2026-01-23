import json
import spacy

nlp = spacy.load("en_core_web_md")

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


#----------- Tokenization -----------#


def tokenize(text):
    
    doc =nlp(text) 
    tokenized_text =[]
    
    for token in doc: 
        
        if not token.is_punct and not token.is_quote and not token.is_space:
            tokenized_text.append(token.lower_)
            
    return tokenized_text


#----------- VOCAB -----------#


def dictionary(list):
    
    token_list =list
    vocab ={PAD_TOKEN:PAD_IDX,UNK_TOKEN:UNK_IDX}
    
    for word in token_list:
        if word not in vocab:
            vocab[word] = len(vocab)
        
    return vocab


def Saved_vocab(vocab,path):
    
    with open(path, "w") as f:
        json.dump(vocab, f)


#----------- Build Sentence From Text -----------#


def build_sentences(text):
    
    doc =nlp(text)
    sentences_list =[]
    
    for sentence in doc.sents:
        sentences_list.append(sentence.text.replace('\n',' ').strip())
    
    return sentences_list 


#----------- Size -----------#


def size(text):
    size_list = []
    for row in text:
    
        size_list.append(len(row))
    return max(size_list)


#----------- Sentence Tokenizer -----------#


def sentece_tokenize(sentence_list):
    
    new_sentences =[]
    
    for sentence in sentence_list:
        
        sent =tokenize(sentence)
        new_sentences.append(sent)
    
    return new_sentences


#----------- Text To Index -----------#


def text_to_index(list_sentences,vocab):
    
    indexed_sentence_list =[]
    
    for sentence in list_sentences:
        
        sentence_to_index =[]
        indexed_sentence_list.append(sentence_to_index)
        
        for token in sentence:
            
            token = vocab[token]
            sentence_to_index.append(token)

    return indexed_sentence_list


#----------- Build training and labelled data-----------#


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


#----------- Padding -----------#


def padding(max_length,train_sequences):
    
    padded_sequences = []
    max_size = max_length
    
    for sequence in train_sequences:
        sequence = sequence[-max_size:]
        padded_sequences.append([0]*((max_size)-len(sequence)) + sequence)
    
    return padded_sequences
