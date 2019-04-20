# This file is responsible for reading the datasets
import enum
import pickle
from utilities import convert_list_by_key, flip_dict

DATA_DIR = '../data/datasets/' # relative path of the directory where we store the datasets
DATASETS = ["BC2GM", "BC4CHEMD", "BC5CDR-IOB", "JNLPBA", "NCBI-disease-IOB"]

# function to preprocess datasets in BIO format
def preprocess_bio (dataset, dataset_split = 'train'):
  paths = [] 
  path = DATA_DIR + dataset + "/" + dataset_split + ".tsv"
  paths.append(path)
  
  if dataset_split ==  'train':
    # we also load the validation data as part of training data
    path = DATA_DIR + dataset + "/" + "devel" + ".tsv"
    paths.append(path)

  ret_sentences = []
  ret_labels = []

  for path in paths:
      with open(path, 'r') as f:
        content = f.readlines()
      
      sentence = []
      tag = []
      for line in content:
        if line != '\n':
          sentence.append(line.split()[0])
          tag.append(line.split()[1])
        else:
          ret_sentences.append(sentence)
          ret_labels.append(tag)
          sentence = []
          tag = []

      
  print ('Dataset {}: ({}) Number of sentences: {}'.
             format(dataset, 
                    dataset_split,
                    len(ret_sentences)
                   )
        )
  return (ret_sentences, ret_labels)


# this function reads through all the datasets, and returns a list of list of sentences, and 
# a list of tags associated with those sentences. 
# It expects an argument to know whether to return test set, or the train set.
# By default it returns the training set, for all the datasets
# It can also return either all the datasets, or just the one specified
def load_datasets(dataset_split = 'train', dataset_index = -1):
    sentences = []
    tags = []
    datasets = []
    if dataset_index == -1:
        datasets = DATASETS # return all
    else:
        if dataset_index < 0 or dataset_index >= len(DATASETS): # invalid index
            raise
        else:
            datasets.append(DATASETS[dataset_index])

    for dataset in datasets:
        sents = ()
        sents = preprocess_bio(dataset, dataset_split)
        if sents:
            sentences.append (sents[0])
            tags.append (sents[1])

    return sentences, tags
