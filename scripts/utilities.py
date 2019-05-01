# this program has all the common utility functions
import json
import os
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Model, Input
from keras.layers import CuDNNLSTM, LSTM, Embedding, TimeDistributed,  Bidirectional, concatenate, Dense, SpatialDropout1D
import random
from nltk import word_tokenize 
from keras_contrib.layers import CRF
from keras.models import load_model
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy, crf_accuracy
import time
import matplotlib.pyplot as plt
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from gensim.models import KeyedVectors
import configparser
import json
config = configparser.ConfigParser()
config.read('config.ini')

TIMESTR = config['DEFAULT']['TIMESTR']
EMBEDDING_FILE_NAME = config['DEFAULT']['EMBEDDING_FILE_NAME']
PRINT_TO_SCREEN = config['DEFAULT'].getboolean('PRINT_TO_SCREEN')
USE_CRF = config['DEFAULT'].getboolean('USE_CRF')
MULTI_OUT =  config['DEFAULT'].getboolean('MULTI_OUT')
MAX_LEN =  config['DEFAULT'].getint('MAX_LEN')
MAX_LEN_CHAR =  config['DEFAULT'].getint('MAX_LEN_CHAR')
EMBEDDING_DIM = config['DEFAULT'].getint('EMBEDDING_DIM')
CHAR_EMBEDDING_DIM = config['DEFAULT'].getint('CHAR_EMBEDDING_DIM')
EPOCHS = config['DEFAULT'].getint('EPOCHS')
EMBEDDING_FILE_NAME = config['DEFAULT']['EMBEDDING_FILE_NAME']
NUM_DS = len (json.loads(config.get("DATASETS","DATASETS")))

timestr = time.strftime("%Y%m%d-%H%M%S") # to differentiate the models, and dictionaries generated during this run.
DICT_PATH = '../data/dict/'
MODEL_PATH = '../model/'
EMBEDDING_FILE = '../word_embeddings/' + EMBEDDING_FILE_NAME
MODEL_NAME = 'model' + TIMESTR + '.h5'
HISTORY_FILE = 'history' + TIMESTR + '.json'

# convert tags to multiple output format
def convert_to_multi_output (ds_y, max_len):
    y = []
    for i, ds in enumerate(ds_y):
            t = []
            for pre in range(0,i):
                len_ds = len(ds_y [pre])
                listofzeros = [[0]*max_len]*len_ds
                t.extend (listofzeros)
            t.extend (ds_y[i])
            for post in range (i+1, len(ds_y)):
                len_ds = len(ds_y [post])
                listofzeros = [[0]*max_len]*len_ds
                t.extend(listofzeros)
            y.append (t)
    return y

# load the model
def load_saved_model():
  custom_objects={'CRF': CRF,
                        'crf_loss': crf_loss,
                        'crf_accuracy': crf_accuracy}
  loaded_model = load_model(MODEL_PATH + MODEL_NAME, custom_objects=custom_objects)

  print("Loaded model from disk")
  
  return loaded_model

# function to load the dictionaries after the run has completed
def load_dict_after(dict_name):
  with open(DICT_PATH + TIMESTR + dict_name, 'r') as fp:
    data = json.load(fp)
  return data

# function to evaluate a pre-trained model on test data-set
def evaluate_on_model (ds_X_word, ds_X_char, ds_y):
    # load padding length
    padding_len = load_dict_after('padding_len.json')
    max_len = min(padding_len['max_len'],MAX_LEN)
    max_len_char = min(padding_len['max_len_char'], MAX_LEN_CHAR)
 
    # load the model in terms of CRF as output layer or Dense as output layer   
    model = load_saved_model()
 
    # prepare the tags in terms of multiple output model, or single output
    y = []
    if USE_CRF and MULTI_OUT:
        y = convert_to_multi_output (ds_y, max_len)
    else:
        for ds in ds_y:
            y.extend(ds)

    X_word = []
    X_char = []

    for x_word in ds_X_word:
        X_word.extend (x_word)

    for x_char in ds_X_char:
        X_char.extend (x_char)
   
    X_word = np.array(X_word,  dtype="float32")
    X_char = np.array(X_char,  dtype="float32")
    
    print (model.metrics_names)       
    
    if MULTI_OUT:
        scores = model.evaluate( [X_word, X_char] , [np.array(y[0], dtype="float32").reshape(len(X_word), max_len,1) , np.array(y[1], dtype="float32").reshape(len(X_word), max_len,1), np.array(y[2], dtype="float32").reshape(len(X_word), max_len,1), np.array(y[3], dtype="float32").reshape(len(X_word), max_len,1), np.array(y[4], dtype="float32").reshape(len(X_word), max_len,1)   ] , verbose=1 )
        print (scores)
    else: # single output
        # get scores for test sets from each data-set
        for i in range(len(ds_X_word)):
            scores = model.evaluate( [ds_X_word[i], ds_X_char[i]] , np.array(ds_y[i], dtype="float32").reshape(len(ds_X_word[i]), max_len,1), verbose=1 )
            print (scores)
    
    if MULTI_OUT:
        test_pred = model.predict ([X_word, X_char])
 
        for i in range (len(ds_X_word)):
            tag2idx = load_dict_after('tag2idx' + str(i) + '.json')
            n_tags  = len (tag2idx)
            idx2tag = flip_dict(tag2idx)
            conv_pred = []
            conv_gold = []
            for sentence_tag in test_pred[i]:
                 p = np.argmax(sentence_tag, axis=-1)
                 p = [idx2tag[tag_idx] for tag_idx in p]
                 conv_pred.append(p)
            for sentence_tag in y[i]:
                 sentence_tag = [idx2tag[tag_idx] for tag_idx in sentence_tag]
                 conv_gold.append(sentence_tag)
          
            print("F1-score: {:.1%}".format(f1_score(conv_gold, conv_pred)))
            print(classification_report(conv_gold, conv_pred))
    else:
       for i in range (len (ds_X_word) ):
            x_word = ds_X_word [i] # all the sentences in word-indexed form for dataset i
            x_char = ds_X_char [i] # all the sentences in character-indexed form for dataset i
            y_sen = ds_y [i]    # all the corresponding tags of the sentences
            y_sen = np.array(y_sen)

            #predict 
            test_pred = model.predict ([np.array(x_word, dtype="float32") , np.array(x_char, dtype="float32")])
            tag2idx = load_dict_after('tag2idx.json')
            n_tags  = len (tag2idx)
            idx2tag = flip_dict(tag2idx)
            conv_pred = [] # list to store the predicted tags, converted from indices
            conv_gold = [] # list to store the actual/ gold tags, converted from indices

            for sentence_tag in test_pred:
                 p = np.argmax(sentence_tag, axis=-1) # for each word, get the tag with maximum probabiliity out of all the possible tags
                 p = [idx2tag[tag_idx] for tag_idx in p] # convert each tag from indice to name
                 conv_pred.append(p)

            for sentence_tag in y_sen:
                 sentence_tag = [idx2tag[tag_idx] for tag_idx in sentence_tag]
                 conv_gold.append(sentence_tag)
          
            print("F1-score: {:.1%}".format(f1_score(conv_gold, conv_pred)))
            print(classification_report(conv_gold, conv_pred))
    
def save_plot(hist, train_str, val_str, metric, ds_num):
    train_metric = hist [train_str]
    val_metric = hist [val_str] 
    
    x_range = range(len(train_metric))

    plt.plot(x_range, train_metric, label='Training ' + metric)
    plt.plot(x_range, val_metric, label='Validation ' + metric)
    
    plt.xlabel('Epochs')
    plt.ylabel(metric)

    plt.title('Dataset ' + ds_num)
    plt.legend()
    plt.savefig(MODEL_PATH + TIMESTR + ds_num + '_' + metric)
    plt.close()

def draw_figures():
  # load the history dictionary
  with open(MODEL_PATH + HISTORY_FILE, 'r') as fp:
    hist = json.load(fp)

  # overall loss, training, and validation
  save_plot (hist, 'loss', 'val_loss', 'loss', 'Complete')
 
  if MULTI_OUT: 
      # plot validation, and training loss, and accuracy across epochs, for each dataset
      for i in range (1, NUM_DS+1):
        # dataset i

        # loss
        loss_str = 'crf_' + str(i) + '_loss'
        val_loss_str = 'val_' + loss_str
        save_plot (hist, loss_str, val_loss_str, 'loss', str(i)) 
      
        # accuracy 
        acc_str = 'crf_' + str(i) + '_crf_accuracy'
        val_acc_str = 'val_' + acc_str
        save_plot (hist, acc_str, val_acc_str, 'accuracy', str(i))
  else: # the validation accuracy is not dataset-specific
       save_plot (hist, 'crf_accuracy', 'val_crf_accuracy', 'accuracy', 'Complete')

# predict
def predict (sentence, loaded_model):
    padding_len = load_dict_after('padding_len.json')
    max_len = min (padding_len['max_len'] , MAX_LEN)
    max_len_char = min(padding_len['max_len_char'], MAX_LEN_CHAR)

    idx2tag = {}
    ds_idx2tag = []
    ds_n_tags = []
    if MULTI_OUT:
        for i in range (NUM_DS):
            tag2idx = load_dict_after('tag2idx' + str(i) + '.json')
            n_tags  = len (tag2idx)
            idx2tag = flip_dict(tag2idx)
            ds_idx2tag.append(idx2tag)
            ds_n_tags.append(n_tags)
    else:
        tag2idx = load_dict_after('tag2idx.json')
        idx2tag = flip_dict(tag2idx)

    # tokenize it, and form a list of list
    tokens = word_tokenize(sentence)
    original_tokens = [] + tokens
    sentences = []
    sentences.append(tokens)
    
    # preprocess: remove digits
    sentences = preprocess_sentences (sentences)
    
    # pad the sentence
    X_word, X_char = prepare_input (sentences, dataset_split='test')

    # load the model
    model = loaded_model

    # predict now
    ds_y_pred = model.predict([np.array(X_word).reshape( (len(X_word), max_len) ), np.array(X_char).reshape((len(X_char), max_len, max_len_char))])
    
    return_tags = [] # this will store the tags associated with the input sentence 
    if MULTI_OUT: 
        for idx, y_pred in enumerate(ds_y_pred): # prediction across each dataset
            i = 0
            p = np.argmax(y_pred[i], axis=-1) # list of predicted tags for each of the max_len words in this sentence
            constrained_p = p [:len(original_tokens)]
            if not np.all(constrained_p == 0): # only if this sentence is relevant for this domain
                if PRINT_TO_SCREEN == True:
                    print("{:15}||{}".format("Word", "Pred"))
                    print(30 * "=")
                return_tag = [] # tags returned by this dataset
                for w, pred in zip(original_tokens, p):
                    return_tag.append(ds_idx2tag[idx][pred])
                    # if all the tags are just PAD we ignore this dataset
                    if PRINT_TO_SCREEN == True:
                        print("{:15}: {}".format(w, ds_idx2tag[idx][pred]))
                return_tags.append(return_tag)
                if PRINT_TO_SCREEN == True:
                    print ('********')
    else:
        i = 0
        p = np.argmax(ds_y_pred[i], axis=-1) # list of predicted tags for each of the max_len words in this sentence
        constrained_p = p [:len(original_tokens)]
        print("{:15}||{}".format("Word", "Pred"))
        print(30 * "=")
        return_tag = [] # tags returned by this dataset
        for w, pred in zip(original_tokens, p):
            return_tag.append(idx2tag[pred] )
            print("{:15}: {}".format(w, idx2tag[pred]))
        return_tags.append (return_tag)
    return original_tokens, return_tags
        
# fit model
def fit_model (model, ds_X_word, ds_X_char, ds_y):
    padding_len = load_dict('padding_len.json')
    max_len = min (padding_len['max_len'] ,MAX_LEN)
    max_len_char = min(padding_len['max_len_char'], MAX_LEN_CHAR)
    
    y = []
    if MULTI_OUT == True:
        # each ds_y[i] should be made
        # expected input: list of [X_word, X_char]
        # expected output: [ [], [], [] ]
        for i, ds in enumerate(ds_y):
            t = []
            for pre in range(0,i):
                len_ds = len(ds_y [pre])
                listofzeros = [[0]*max_len]*len_ds
                t.extend (listofzeros)
            t.extend (ds_y[i])
            for post in range (i+1, len(ds_y)):
                len_ds = len(ds_y [post])
                listofzeros = [[0]*max_len]*len_ds
                t.extend(listofzeros)
            y.append (t) 
    else:
        for ds in ds_y:
            y.extend(ds)   
    
    X_word = []
    X_char = []
    
    for x_word in ds_X_word:
        X_word.extend (x_word)
        
    for x_char in ds_X_char:
        X_char.extend (x_char)
    
    # convert to np arrays
    X_word = np.array(X_word,  dtype="float32")    
    X_char = np.array(X_char,  dtype="float32")
    
    # shuffle 
    indices = np.arange(X_word.shape[0])
    np.random.shuffle(indices)
    X_word = X_word[indices]
    X_char = X_char[indices]
   
    if MULTI_OUT == True: 
        b = np.array (y)
        for i, bucket in enumerate(y):
            bucket = np.array(bucket, dtype="float32")
            bucket = bucket[indices]
            b[i] = bucket
       
        if (b.shape[0] == 1): # single dataset
            history = model.fit([X_word, X_char],
                    [b[0].reshape(len(X_word), max_len, 1)],
                    batch_size=32, epochs=EPOCHS, validation_split=0.1, verbose=1, shuffle=True)
        else:
            history = model.fit([X_word, X_char],
                        [b[0].reshape(len(X_word), max_len, 1), b[1].reshape(len(X_word), max_len, 1), b[2].reshape(len(X_word), max_len, 1), b[3].reshape(len(X_word), max_len, 1), b[4].reshape(len(X_word), max_len, 1)],
                        batch_size=32, epochs=EPOCHS, validation_split=0.1, verbose=1)
    else:
        y = np.array(y)
        y = y[indices]
        history = model.fit([X_word, X_char.reshape(len(X_char), max_len, max_len_char)], y.reshape(len(X_word), max_len, 1), batch_size=32, epochs=EPOCHS, validation_split=0.1, verbose=1, shuffle=True)
      
        
    model.save(MODEL_PATH + 'model' + timestr + '.h5')
    
    # dump the accuracy, and loss metrics
    with open(MODEL_PATH + 'history' + timestr + '.json', 'w') as fp:
        json.dump(history.history, fp)

    print("Saved model to disk")
    

# prepare model
def prepare_model (embedding_matrix, num_ds):
  
  # load dictionaries
  word2idx = load_dict('word2idx.json')
  char2idx = load_dict('char2idx.json')
  
  ds_tag2idx = []
  ds_n_tags = []
  
  for i in range (num_ds):
    tag2idx = load_dict('tag2idx' + str(i) + '.json')
    n_tags  = len (tag2idx)
    ds_tag2idx.append(tag2idx)
    ds_n_tags.append(n_tags)

  padding_len = load_dict('padding_len.json')
 
  max_len = min(padding_len['max_len'], MAX_LEN)
  max_len_char = min(padding_len['max_len_char'], MAX_LEN_CHAR)
  
  n_words = len (word2idx)
  n_chars = len (char2idx)
    
  word_in = Input(shape=(max_len,))
  
  # change each word in sentence to a pre-trained embedding
  emb_word = Embedding(n_words, EMBEDDING_DIM,mask_zero=True,
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=False) (word_in)

  # each word is max_len_char long, and each sentence is max_len long 
  char_in = Input(shape=(max_len, max_len_char,))

  # change each char to an embedding vector
  emb_char = TimeDistributed(Embedding(input_dim=n_chars, output_dim= CHAR_EMBEDDING_DIM,
                             input_length=max_len_char, mask_zero=True))(char_in)

  # learn a word embedding from the character via a bi-LSTM
  char_enc = TimeDistributed(Bidirectional(LSTM(units=50, return_sequences=False,
                                  recurrent_dropout=0.5)))(emb_char)

  # concatenate the word embedding with the embedding derived from characters
  x = concatenate([emb_word, char_enc])

  x = SpatialDropout1D(0.3)(x)

  # encode the entire sentence, and learn context between words via bi-LSTM
  main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                                 recurrent_dropout=0.6))(x)
  all_tag2idx = load_dict('tag2idx.json')
  all_n_tags =  len(all_tag2idx)

  if USE_CRF:
    dense = TimeDistributed(Dense(50, activation="relu"))(main_lstm)
    outs =[]
    if MULTI_OUT: #push multiple CRF models
        for i in range (num_ds):
            crf = CRF(ds_n_tags[i], sparse_target=True)  # CRF layer
            out = crf(dense)  # output
            outs.append (out)
    else:
        crf = CRF(all_n_tags, sparse_target=True)  # CRF layer
        out = crf(dense)  # output
        outs.append (out)

    model = Model(inputs=[word_in, char_in], outputs= outs)
    model.compile(optimizer="rmsprop", loss=crf_loss, metrics=[crf_accuracy])
  else:
    out = TimeDistributed(Dense(all_n_tags, activation="softmax"))(main_lstm)
    model = Model([word_in, char_in], out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"]) 

  model.summary()
  
  return model

# For a given list having values as the keys in dictionary dict, 
# this function changes the list to be the values corresponding to the keys
def convert_list_by_key (dict, lst):
    return [[ dict[idx] for idx in sents] for sents in lst]

# return a flipped dictionary where values become keys
def flip_dict (dict):
    return {val:key for key, val in dict.items()}

# function to remove digits from sentences, and replace them with DIGIT
# Example: '12' => 'DIGITDIGIT'; 'G1B1' => 'GDIGITBDIGIT'
def remove_digits (sents):
  for i, sent in enumerate(sents):
    for j, word in enumerate (sent):
      if word.isdigit():
        sents[i][j] = len(word)*"DIGIT"
      else:
        charInside = 0
        for char in word:
          if char.isdigit():
            charInside = 1
            break
        if charInside == 1:
          newWord = ""
          for char in word:
            if not char.isdigit():
              newWord += char
            else:
              newWord += "DIGIT"
          sents[i][j] = newWord
  return sents

# function to lowercase all the words in a sentence.
# Accepts a list of sentences, where each sentence is a list of words
def lowercase_data(sentences):
    for i, sent in enumerate(sentences):
        for j, word in enumerate(sent):
            if not word.islower():
                sentences[i][j] = word.lower()
    return sentences


# function to preprocess sentences, it does the following:
# a) replace any digits by the word DIGIT b) lower-case the entire sentence
def preprocess_sentences (sentences):
    # replace digits with the word DIGIT
    sentences = remove_digits (sentences)

    # lowercase the entire sentences
    #sentences = lowercase_data (sentences)

    return sentences

def create_vocab_tags(ds_tags):
  # tags
  all_tags_vocab = set()
  for i, tags in enumerate(ds_tags):
    tags_vocab = set()
    for tag in tags:
        tags_vocab.update(tag)
        all_tags_vocab.update(tag)

    tags_vocab = list(tags_vocab)
  
    tag2idx = {t: i + 1 for i, t in enumerate(tags_vocab)}
    tag2idx["PAD"] = 0 # "PAD" tag for "PAD" words in padded sentences

    print ('Number of unique tags in training set (including PAD) for dataset '+str(i)+': ', len (tag2idx))
 
    # dump the dictionary
    with open(DICT_PATH + timestr + 'tag2idx'+str(i)+'.json', 'w') as fp:
        json.dump(tag2idx, fp)

  # vocab for consolidated tags 
  all_tags_vocab = list(all_tags_vocab)
  tag2idx = {t: i + 1 for i, t in enumerate(all_tags_vocab)}
  tag2idx["PAD"] = 0 # "PAD" tag for "PAD" words in padded sentences

  print ('Number of unique tags in training set (including PAD) across all datasets ', len (tag2idx))

  # dump the dictionary
  with open(DICT_PATH + timestr + 'tag2idx.json', 'w') as fp:
        json.dump(tag2idx, fp)



# function to create vocabulory for words, characters
def create_vocab(sentences):
  # words
  words_vocab = set()
  for sent in sentences:
    words_vocab.update(sent)

  words_vocab = list(words_vocab)
  
  word2idx = {w: i + 2 for i, w in enumerate(words_vocab)} # first 2 indices saved for special words
  word2idx["UNK"] = 1 # unknown word
  word2idx["PAD"] = 0 # PAD-ding word  for sentences
 
  print ('Number of unique words in training set (including PAD, UNK): ', len (word2idx))
   
  # dump the dictionary
  with open(DICT_PATH + timestr + 'word2idx.json', 'w') as fp:
    json.dump(word2idx, fp)

  # characters
  chars_vocab = set([w_i for w in words_vocab for w_i in w])
  
  char2idx = {c: i + 2 for i, c in enumerate(chars_vocab)}
  char2idx["UNK"] = 1 # unknown new char index
  char2idx["PAD"] = 0 # padded char index
 
  print ('Number of unique characters in training set (including PAD, UNK): ', len(char2idx))
 
   
  # dump the dictionary
  with open(DICT_PATH + timestr  + 'char2idx.json', 'w') as fp:
    json.dump(char2idx, fp)
  
  max_len = max([len (s) for s in sentences])
  print ('Maximum sentence length in training set: ', max_len)

  max_len_char = max([max ([len(w) for w in s]) for s in sentences])
  print ('Maximum word length in training set: ', max_len_char)

  padding_len = {'max_len': max_len, 'max_len_char': max_len_char}

  # dump max_len, and max_len_char
  with open(DICT_PATH + timestr + 'padding_len.json', 'w') as fp:
    json.dump(padding_len, fp) 

# function to load the dictionaries
def load_dict(dict_name):
  with open(DICT_PATH + timestr +dict_name, 'r') as fp:
    data = json.load(fp)
  return data


# function to replace words in sentence with indices, and pads to max sentence length.
# Also replaces characters in words with indices, and pads to max word length
def prepare_input (sentences, dataset_split = 'train'):
  # load dictionaries
  if dataset_split == 'train':
      word2idx = load_dict('word2idx.json')
      char2idx = load_dict('char2idx.json')
      padding_len = load_dict('padding_len.json')
  else:
    # test split; load dictionaries from the pre-trained data
      word2idx = load_dict_after('word2idx.json')
      char2idx = load_dict_after('char2idx.json')
      padding_len = load_dict_after('padding_len.json')
    

  max_len = min (padding_len['max_len'], MAX_LEN)
  max_len_char = min( padding_len['max_len_char'], MAX_LEN_CHAR )

  # replace words with indices
  X_word = []
  for sentence in sentences:
    sen = []
    for w in sentence:
        try:
            sen.append(word2idx[w])
        except:
            sen.append(word2idx["UNK"])
    X_word.append (sen)        
    
  # pad each sentence to form same length
  X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=word2idx["PAD"], padding='post', truncating='post')
  
  # replace characters in words with indices
  X_char = []
  for sentence in sentences:
    sen_idx = []
    for word in sentence:
        word_idx = [char2idx.get(c) if c in char2idx else char2idx.get("UNK") for c in word]
        sen_idx.append (word_idx)
    sen_idx = pad_sequences(maxlen=max_len_char, sequences=sen_idx, value=char2idx["PAD"], padding='post', truncating='post')
    X_char.append (sen_idx) 
  
  X_char = pad_sequences(maxlen=max_len, sequences=X_char, value=word2idx["PAD"], padding='post', truncating='post')

  return X_word, X_char

# For each dataset, function to replace tags with indices, and pad it to max sentence length
def prepare_tags (ds_tags, dataset_split = 'train'):
  ds_y = []
  if dataset_split == 'train':
    all_tag2idx =  load_dict('tag2idx.json')
  else: 
    all_tag2idx =  load_dict_after('tag2idx.json')
 
  for i, tags in enumerate(ds_tags):
    # load dictionary
    if MULTI_OUT:
        if dataset_split == 'train':
            tag2idx = load_dict('tag2idx' + str(i) + '.json')
        else:
            tag2idx = load_dict_after('tag2idx' + str(i) + '.json')
    else:
        tag2idx = all_tag2idx

    if dataset_split == 'train':
        padding_len = load_dict('padding_len.json')
    else:
        padding_len = load_dict_after('padding_len.json')

    max_len = min( padding_len['max_len'], MAX_LEN)
    
    y = [] 
    try:
        y = [ [tag2idx[tag] if tag in tag2idx else tag2idx.get("O") for tag in sen_tags] for sen_tags in tags]
    except:
        print ('ERROR: tag2idx convertion failed! There is a tag in data, not covered in training set.')
    y = pad_sequences(maxlen=max_len, sequences=y, value=tag2idx["PAD"], padding='post', truncating='post')
    
    ds_y.append(y)
    
  return ds_y

# function to create an embedding matrix for each of the unique word in training data, using
# pre-trained embeddings trained on much larger training data.
def load_embedding_matrix ():
  # try to load the embedding matrix first if it already exists
  try:
    word2idx = load_dict ('word2idx.json')

    embedding_matrix = np.load(DICT_PATH + timestr + 'embedding_matrix.npy')
    if (embedding_matrix.shape[0] != len (word2idx)):
        raise # the embedding matrix might have been loaded for some other dataset 
  except:
    embedding_index = {}
    print ('embedding matrix is not populated from pre-trained embeddings. Populating now, this will take time...')
   
    #f = open(EMBEDDING_FILE,  'r', encoding='utf-8', newline='\n', errors='ignore')
    #for line in f:
     #   values = line.split()
  #      word = values[0]
   #     coefs = np.asarray(values[1:], dtype='float32')
    #    embedding_index[word] = coefs
    #f.close()
    #print('Loaded %s word vectors.' % len(embedding_index))
 
    wv = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    
    # create embedding matrix from the embedding index now.
    embedding_matrix = (np.random.rand (len(word2idx), EMBEDDING_DIM)-0.5)/5.0
    for word, i in word2idx.items():
        try:
            embedding_matrix[i] = wv[word]
        except:
            continue
        #embedding_vector = embedding_index.get(word)
        #if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
         #   embedding_matrix[i] = embedding_vector
    print ("Embedding matrix created.\n")
    
    # save it for future purpose
    np.save(DICT_PATH + timestr  + 'embedding_matrix', embedding_matrix)  

  return embedding_matrix
