# BioNER - CSCE-633
Biomedical Named Enity Recogntion: Course project for CSCE 633.

Uses character-level BiLSTM layer, and then Bi-LSTM-CRF layers to do NER for biomedical entities.

## How to run? 
1. Clone this repo using: 
```
    git clone git@github.tamu.edu:rssaini/CSCE-633.git
```

2. Use the below command to replicate the provided  conda environment to satisfy the requirements for running the project. Within the project root folder:
```
    cd scripts/
    conda env create -f environment.yml
```

3. Step 2 will create a conda environment with name 'keras-gpu-2.2.2'. Now activate the environment via:
```
    conda activate keras-gpu-2.2.2

```
4. Once, inside the conda environment, you can run the code. See the next section to see what to run to achieve what.

5. Since the pre-trained word embeddings are large, they are not provided as part of this repo. Download the word embeddings first via:

```
mkdir word_embeddings
cd word_embeddings/
wget http://evexdb.org/pmresources/vec-space-models/PubMed-w2v.bin
```

## What to run?

1. All the datasets are stored in data/datasets/, all the trained models, and history files get stored in model/, all the dictionaries, and embedding matrices get stored in data/dict/, word embeddings are stored in word\_embeddings

2. The code is capable of running in different modes. The only file to make changes is:
```
scripts/config.ini
```

3. The default configuration provided runs a multi-task (implying trained on all the datasets, and tags), with CRF, single-output-layer model (means there is no branching in output layer, and it uses CRF as the output layer): 

### to reproduce the results on test set
```
cd scripts/
python evaluate.py
```
### to run a GUI to predict slots on sentences
```
cd scripts/
python new_gui.py
# type in related sentences, you can refer into data/datasets/ to get example sentences
# example sentence: Number of glucocorticoid receptors in lymphocytes and their sensitivity	to hormone action.
```

### to predict sentences on console
```
cd scripts/
python predict.py
# type in related sentences, you can refer into data/datasets/ to get example sentences
```

### training a new model
```
# make sure you have created the word_embeddings folder, and have the embedding file downloaded in it
cd scripts/
python train.py
# the model gets saved in model/. Make note of the timestr appended to the model name, 
# example: if model name is  model20190426-042026.h5, then 20190426-042026 is the timestr.
```

### running evaluate and test on a newly trained model model
We need to provide the timestr of the newly trained model in the config.ini file
```
vi scripts/config.ini
TIMESTR= # put the value of timestr corresponding to the new model
```

### trying different variations
#### training  a single task model: 
```
vi scripts/config.ini
DATASET_INDEX = 0 # change it to the index of the DATASET for which you want to create a single task model. 
```

#### training a model using multiple outputs
```
vi scripts/config.ini
MULTI_OUT = True
```

### training a model with Dense as output layer, instead of CRF
```
vi scripts/config.ini
# For Dense, single output
MULTI_OUT = False
USE_CRF = False

# For Dense, multi output
MULTI_OUT = True
USE_CRF = False
```
