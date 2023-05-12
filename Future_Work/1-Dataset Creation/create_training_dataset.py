from collections import defaultdict
import json
import numpy as np
import pandas as pd
import pickle

# Define a loader function for SNLI jsonl format
def load_snli(fpaths):
    sa, sb, lb = [], [], []
    fpaths = np.atleast_1d(fpaths)
    for fpath in fpaths:
        with open(fpath) as fi:
            for line in fi:
                sample = json.loads(line)
                sa.append(sample['sentence1'])
                sb.append(sample['sentence2'])
                lb.append(sample['gold_label'])
    return sa, sb, lb

# For each unique anchor we create an ID and an entry containing the anchor, entailment and contradiction samples_tr.
# The anchors lacking at least one sample of each class are filtered out.
def prepare_snli(sa, sb, lb):
    
    classes = {"entailment", "contradiction"}
    anc_to_pairs = defaultdict(list)
    filtered = {}
    skipped = 0
    anchor_id = 0

    for xa, xb, y in zip(sa, sb, lb):
        anc_to_pairs[xa].append((xb, y))

    for anchor, payload in anc_to_pairs.items(): 
        
        filtered[anchor_id] = defaultdict(list)
        filtered[anchor_id]["anchor"].append(anchor)
        
        labels = set([t[1] for t in payload])
        
        if len(labels&classes) == len(classes):
            for text, label in payload:
                filtered[anchor_id][label].append(text)
            anchor_id += 1
        else:
            skipped += 1
            
    print("Loaded: {} \nSkipped: {}".format(anchor_id, skipped))
        
    return filtered


# Load the SNLI and MNLI datasets.
train_data = ["./snli_1.0/snli_1.0_train.jsonl", "./multinli_1.0/multinli_1.0_train.jsonl"]
test_data = ["./snli_1.0/snli_1.0_test.jsonl", "./multinli_1.0/multinli_1.0_dev_matched.jsonl"]

tr_a, tr_b, tr_l = load_snli(train_data)
ts_a, ts_b, ts_l = load_snli(test_data)

fd_tr = prepare_snli(tr_a, tr_b, tr_l)
fd_ts = prepare_snli(ts_a, ts_b, ts_l)



# Create metadata_tr and samples_tr for training file
metadata = {}
samples = []
train_indecies = []
sentence_id = 0
rejected = 0
for idx in range(len(fd_tr)):
    if(len(fd_tr[idx]['anchor']) == 0 or len(fd_tr[idx]['neutral']) == 0 or len(fd_tr[idx]['entailment']) == 0 or len(fd_tr[idx]['contradiction']) == 0):
        rejected += 1
        continue
    anchor = fd_tr[idx]['anchor'][0]
    neutral = fd_tr[idx]['neutral'][0]
    entailment = fd_tr[idx]['entailment'][0]
    contradiction = fd_tr[idx]['contradiction'][0]
    
    # metadata Part
    metadata.update({str(sentence_id) : {'paper_id' : str(sentence_id), 'abstract' : '', 'title' : anchor}})
    metadata.update({str(sentence_id + 1) : {'paper_id' : str(sentence_id + 1), 'abstract' : '', 'title' : neutral}})
    metadata.update({str(sentence_id + 2) : {'paper_id' : str(sentence_id + 2), 'abstract' : '', 'title' : entailment}})
    metadata.update({str(sentence_id + 3) : {'paper_id' : str(sentence_id + 3), 'abstract' : '', 'title' : contradiction}})
    
    # samples_tr Part
    hard_neg_sample = [str(sentence_id), (str(sentence_id + 2), 5), (str(sentence_id + 1), 1)]
    samples.append(hard_neg_sample)
    easy_neg_sample = [str(sentence_id), (str(sentence_id + 2), 5), (str(sentence_id + 3), float("-inf"))]
    samples.append(easy_neg_sample)
    
    # training indecies
    train_indecies.append(str(sentence_id))
    
    # Increment element
    sentence_id += 4

# Print number of rejected ones
print('\nRejected: ', rejected)

# Read samples_tr & Dump metadata_tr into Json
print(f'\nThis is a sample from the samples:\n{samples[0]}\n{samples[-1]}')

# print metadata length
print('\nlength of metadata after processing the training data: ', len(metadata))

# Create metadata_ts and samples_ts for training file
test_indecies = []
sentence_id = len(metadata)
rejected = 0
for idx in range(len(fd_ts)):
    if(len(fd_ts[idx]['anchor']) == 0 or len(fd_ts[idx]['neutral']) == 0 or len(fd_ts[idx]['entailment']) == 0 or len(fd_ts[idx]['contradiction']) == 0):
        rejected += 1
        continue
    anchor = fd_ts[idx]['anchor'][0]
    neutral = fd_ts[idx]['neutral'][0]
    entailment = fd_ts[idx]['entailment'][0]
    contradiction = fd_ts[idx]['contradiction'][0]
    
    # metadata_tr Part
    metadata.update({str(sentence_id) : {'paper_id' : str(sentence_id), 'abstract' : '', 'title' : anchor}})
    metadata.update({str(sentence_id + 1) : {'paper_id' : str(sentence_id + 1), 'abstract' : '', 'title' : neutral}})
    metadata.update({str(sentence_id + 2) : {'paper_id' : str(sentence_id + 2), 'abstract' : '', 'title' : entailment}})
    metadata.update({str(sentence_id + 3) : {'paper_id' : str(sentence_id + 3), 'abstract' : '', 'title' : contradiction}})
    
    # samples_tr Part
    hard_neg_sample = [str(sentence_id), (str(sentence_id + 2), 5), (str(sentence_id + 1), 1)]
    samples.append(hard_neg_sample)
    easy_neg_sample = [str(sentence_id), (str(sentence_id + 2), 5), (str(sentence_id + 3), float("-inf"))]
    samples.append(easy_neg_sample)
    
    # testing indecies
    test_indecies.append(str(sentence_id))
    
    # Increment element
    sentence_id += 4

# Print number of rejected ones
print('\nRejected: ', rejected)

# Read samples_tr & Dump metadata_tr into Json
print(f'\nThis is a sample from the samples:\n{samples[0]}\n{samples[-1]}')

# Create validation indecies
val_indecies = train_indecies[(len(train_indecies) - 5000) :]
train_indecies = train_indecies[: (len(train_indecies) - 5000)]
print('\nlength of train set: ', len(train_indecies))
print('length of test set: ', len(test_indecies))
print('length of val set: ', len(val_indecies))

# Save train, test, val files
with open('train.txt', 'w') as f:
    for line in train_indecies:
        f.write(line)
        f.write('\n')
        
with open('test.txt', 'w') as f:
    for line in test_indecies:
        f.write(line)
        f.write('\n')
        
with open('val.txt', 'w') as f:
    for line in val_indecies:
        f.write(line)
        f.write('\n')

# Save metadata
with open("new_metadata.json", "w") as write_file:
    json.dump(metadata, write_file)

# Convert samples to dict
samples_dict = {}
idx = '0'
one_idx_list = []
for sample in samples:
    if sample[0] == idx:
        one_idx_list.append(sample)
    else:
        samples_dict.update({idx : one_idx_list})
        idx = sample[0]
        one_idx_list = [sample]

# Save samples
with open('samples.pickle', 'wb') as output:
    pickle.dump(samples_dict, output)


# Load samples_tr
with open('samples.pickle', 'rb') as data:
    samples_tr = pickle.load(data)
    
print('\nReading from samples pickle:')
for idx, i in enumerate(samples_tr):
    print(samples_tr[i])
    if idx == 5:
        break

print('\nDONE!!\n')



