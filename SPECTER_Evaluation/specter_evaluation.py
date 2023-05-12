from __future__ import absolute_import, division, unicode_literals
import logging
import sys

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data/'

# import Senteval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

import numpy as np
from transformers import AutoTokenizer, AutoModel

# load SPECTER model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter')

def prepare(params, samples):
    return

def batcher(params, batch):
    batch = [' '.join(s) for s in batch]
    
    # concatenate title and abstract        
    batch = [sent + tokenizer.sep_token + '' for sent in batch]
    
    # preprocess the input
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
    result = model(**inputs)
    
    # take the first token in the batch as the embedding
    embeddings = np.array(result.last_hidden_state[:, 0, :].detach().cpu().numpy().tolist())
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 512}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":

    se = senteval.engine.SE(params_senteval, batcher, prepare)
                        
    transfer_tasks = [#'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      #'CR', 'MPQA', 'MR', 
                      #'SUBJ',
                      'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark']
                      #'Length', 'WordContent', 'Depth', 'TopConstituents',
                      #'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      #'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)
