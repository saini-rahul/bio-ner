# this program will train the datasets on the model, and save the fitted model

from load_data import load_datasets
from utilities import preprocess_sentences, create_vocab, create_vocab_tags, prepare_input, prepare_tags, load_embedding_matrix, prepare_model, fit_model

CLIP_SEN = 20000


print ("B: load")
# STEP 1: load datasets
# ds_sentences[i] contains the list of sentences for dataset i, and ds_tags[i] the corresponding tags
ds_sentences, ds_tags = load_datasets () # by default, this will load the training datasets

print ("B: preprocess")

# STEP 2: pre-process the data
# replace digits across each dataset
for i, sentences in enumerate(ds_sentences): # iterate over each dataset
    sentences = sentences[:10000]
    ds_tags[i] = ds_tags[i][:10000]
    ds_sentences[i] = preprocess_sentences (sentences)

print ("B: create_vocab")

# STEP 3: create and save vocabulary dictionaries for words, and characters
# as well as the length of each sentence, and length of each word
consolidated_sen = []
for sentences in ds_sentences:
    consolidated_sen.extend (sentences)
create_vocab (consolidated_sen) # we will have just one common vocab across all datasets

print ("B: tags vocab")

# STEP 4: create separate vocab of tags across each dataset
create_vocab_tags (ds_tags)

# STEP 5: For each dataset, convert each sentence from a list of words to list of indices,
# and pad it to be of max_len size, with each word being max_len_char size
print ('Padding sentences, and words. This will take some time...')

ds_X_word = [] 
ds_X_char = []

for sentences in ds_sentences: 
    X_word, X_char = prepare_input (sentences)
    ds_X_word.append (X_word)
    ds_X_char.append (X_char)

print ('Padding done!')


# STEP 6: For each dataset, prepare tags by converting them to indices
ds_y = prepare_tags (ds_tags)

# STEP 7: prepare, and load pre-trained embedding matrix
embedding_matrix = load_embedding_matrix ()

# STEP 7: prepare model
model = prepare_model (embedding_matrix, len(ds_sentences))

# STEP 8: FEED THE MONSTER!!
print ('Training begins...')
fit_model(model, ds_X_word, ds_X_char, ds_y)
