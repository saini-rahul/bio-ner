# this program will evaluate the given model. If the model is multi-task learned it will evaluate accuracy over each dataset separately,
# if it is single-task then we just evaluate over the dataset for which it corresponds to.
# important note before running this script: 
# 1) Put the correct pre-trained model name in utilities.py MODEL_NAME 
# 2) Put the correct TIMESTR to match the dictionaries generated for this model

from load_data import load_datasets
from utilities import preprocess_sentences, create_vocab, create_vocab_tags, prepare_input, prepare_tags, evaluate_on_model
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

DATASET_INDEX = config['DEFAULT'].getint('DATASET_INDEX') # this controls whether we want to load sentences from all datasets (set it to -1), or we want a specific dataset, identified by th


# STEP 1: load test datasets
# ds_sentences[i] contains the list of sentences for dataset i, and ds_tags[i] the corresponding tags
ds_sentences, ds_tags = load_datasets (dataset_split = 'test', dataset_index = DATASET_INDEX) # we want to load the test data files, and since dataset_name is not provided it will load the test files for all the datasets.

# STEP 2: pre-process the data
# replace digits, and lowercase the words across each dataset
for i, sentences in enumerate(ds_sentences): # iterate over each dataset
    sentences = sentences[:10000] # we only test the first 10k sentences, as we trained on 20k sentences
    ds_tags[i] = ds_tags[i][:10000]
    ds_sentences[i] = preprocess_sentences (sentences)

# STEP 5: For each dataset, convert each sentence from a list of words to list of indices,
# and pad it to be of max_len size, with each word being max_len_char size
print ('Padding sentences, and words. This will take some time...')

ds_X_word = [] 
ds_X_char = []

for sentences in ds_sentences: 
    X_word, X_char = prepare_input (sentences, dataset_split = 'test')
    ds_X_word.append (X_word)
    ds_X_char.append (X_char)

print ('Padding done!')

# STEP 6: For each dataset, prepare tags by converting them to indices
ds_y = prepare_tags (ds_tags, dataset_split = 'test')

# STEP 7: evaluate on model
evaluate_on_model (ds_X_word, ds_X_char, ds_y)
