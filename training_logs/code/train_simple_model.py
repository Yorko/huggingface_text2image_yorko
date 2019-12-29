#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install pytorch_pretrained_biggan
#!pip install nltk


# In[1]:


import os
import sys
sys.path.append('..')
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from pytorch_pretrained_biggan.utils import IMAGENET, one_hot_from_names
from transformers import AutoTokenizer, AutoModel
from pytorch_pretrained_biggan import BigGAN

import torch
from torch.utils.data import Dataset, DataLoader


# ### Load COCO captions

# In[2]:


#!mkdir -p ../data/coco_captioning/


# In[3]:


coco_data = load_coco_data(base_dir='../data/coco_captioning/')


# In[4]:


coco_data['train_captions'].shape, coco_data['val_captions'].shape


# In[5]:


train_captions = [decode_captions(coco_data['train_captions'][i], coco_data['idx_to_word'])
                  for i in range(coco_data['train_captions'].shape[0])]
val_captions = [decode_captions(coco_data['val_captions'][i], coco_data['idx_to_word'])
                  for i in range(coco_data['val_captions'].shape[0])]


# In[6]:


train_captions[:5]


# ### ImageNet classes
# 
# The code is reused from `multimodal-code-exercise/prepare_data.py` by HuggingFace.

# In[11]:


class_to_synset = dict((v, wn.synset_from_pos_and_offset('n', k)) 
                       for k, v in IMAGENET.items())


# In[12]:


lm_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')


# In[13]:


words_dataset = {}
all_words = set()
for i, synset in tqdm(class_to_synset.items()):
    current_synset = synset
    while current_synset:
        for lemma in current_synset.lemmas():
            name = lemma.name().replace('_', ' ').lower()
            if name in all_words:
                continue  # Word is already assigned
            if lm_tokenizer.convert_tokens_to_ids(name) != lm_tokenizer.unk_token_id:
                # Word is in Bert tokenizer vocabulary
                words_dataset[i] = name
                all_words.add(name)
                current_synset = False # We're good
                break
        if current_synset and current_synset.hypernyms():
            current_synset = current_synset.hypernyms()[0]
        else:
            current_synset = False


# In[14]:


len(words_dataset)


# ### Filter and process COCO captions

#  - Replace \<\UNK\> with [UNK]
#  - strip start and end tokens
#  - select only captions with one ImageNet word

# In[36]:


def filter_and_process_captions(captions, imagenet_words):
    
    result, labels = [], []
    for caption in captions:
        common_words = list(set(caption.split()).intersection(imagenet_words))

        if len(common_words) == 1:
            result.append(caption.replace('<START>', '').replace('<END>', '')                      .replace('<UNK>', '[UNK]').strip())
            labels.append(common_words[0])
    return result, labels


# In[37]:


train_captions, train_labels = filter_and_process_captions(train_captions, all_words)
val_captions, val_labels = filter_and_process_captions(val_captions, all_words)


# In[38]:


len(train_captions), len(val_captions)


# In[17]:


train_captions[:5]


# In[39]:


train_labels[:5]


# ### Explore COCO captions

# In[19]:


imagenet_class_distibution_coco = {word: 0 for word in all_words}
num_imagenet_classes_per_caption = []
caption_len = []

for caption in tqdm(train_captions):
    common_words = set(caption.split()).intersection(all_words)
    
    num_imagenet_classes_per_caption.append(len(common_words))
    caption_len.append(len(caption.split()))
    for word in common_words:
        imagenet_class_distibution_coco[word] += 1


# Train captions are 15 max in length, 10 in median

# In[20]:


pd.Series(caption_len).describe()


# From now on, this 15 will be maximal sequence length

# In[21]:


MAX_SEQ_LEN = 15


# Many (433) of ImageNet classes are not there in train COCO captions. Can be fixed by augmenting captions.

# In[23]:


sorted(imagenet_class_distibution_coco.items(), key=lambda pair: -pair[1])


# In[24]:


sum([1 for (k, v) in imagenet_class_distibution_coco.items() if v == 0])


# ## Save train and validation COCO captions

# In[25]:


with open('../data/train_coco_captions_one_class.pkl', 'wb') as f:
    pickle.dump(train_captions, f)
with open('../data/val_coco_captions_one_class.pkl', 'wb') as f:
    pickle.dump(val_captions, f)


# ## Extend COCO captions with simple patterns

# In[ ]:


patterns = ['I saw a <WORD>',
            'we can notice a <WORD>',
            'a <WORD> is there',
           'this is the best <WORD> ever seen']


# In[ ]:





# In[ ]:





# ### LM hidden states (mapping model input)

# In[26]:


lm_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
lm_model = AutoModel.from_pretrained('distilbert-base-uncased')


# In[27]:


def encode_text_input(text, tokenizer, language_model, max_seq_length=MAX_SEQ_LEN):
    
    pad_vid = lm_tokenizer.vocab["[PAD]"]
    
    # encoding the text
    text_encoded = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
    
    # padding short texts
    true_seq_length = text_encoded.size(1)
    pad_size = max_seq_length - true_seq_length
    if pad_size > 0:
        pad_ids = torch.Tensor([pad_vid] * pad_size).long().view([-1, pad_size])
        text_encoded_padded_or_trimmed = torch.cat((text_encoded, pad_ids), dim=1)
    else:
        text_encoded_padded_or_trimmed = text_encoded[:max_seq_length]
    
    with torch.no_grad():
        # Last layer hidden-states are the first output of 
        # Transformers' library models output tuple
        lm_hidden_states = language_model(text_encoded_padded_or_trimmed)[0]
        
        # Keep first example in the batch - output shape (seq length, hidden size)
        lm_hidden_states_first_example = lm_hidden_states[0] 
    
    return lm_hidden_states_first_example


# Example usage

# In[30]:


res = encode_text_input(text='A dog is faster than a cat',
                       tokenizer=lm_tokenizer, language_model=lm_model)


# In[31]:


res.size()


# In[32]:


train_captions[6]


# In[33]:


encode_text_input(text=train_captions[6],
                       tokenizer=lm_tokenizer, language_model=lm_model)


# ### BigGan class embeddings (mapping model target)

# In[34]:


gan_model = BigGAN.from_pretrained('biggan-deep-128')


# In[68]:


def create_biggan_embeggings(text, label, gan_model, imagenet_words, max_seq_len):
    
    result = []
    tokens = text.split()
    
    if len(tokens) < max_seq_len:
        tokens += ['[PAD]' for _ in range(max_seq_len - len(tokens))]
    else:
        tokens = tokens[:max_seq_len]
    
    for token in tokens:
        try:
            class_vector = torch.tensor(one_hot_from_names(label)) 
        except AssertionError:
             class_vector = torch.zeros([1, 1000])
        result.append(gan_model.embeddings(class_vector))
        
    with torch.no_grad():   
        result = torch.cat(result)
    
    return result


# Example usage

# In[69]:


res = create_biggan_embeggings(text='A dog is faster than a cat',
                               label='dog',
                               gan_model=gan_model,
                               imagenet_words=all_words, 
                               max_seq_len=15)


# In[70]:


res.size()


# In[71]:


train_captions[28], train_labels[28]


# In[72]:


create_biggan_embeggings(text=train_captions[28],
                         label=train_labels[28],
                               gan_model=gan_model,
                               imagenet_words=all_words, 
                               max_seq_len=15)


# ### Dataset to train the mapping model

# In[73]:


class MappingDataset(Dataset):
    """
    """
    def __init__(self,
                 captions,
                 labels,
                 imagenet_words,
                 tokenizer,
                 language_model,
                 gan_model, 
                 max_seq_length=MAX_SEQ_LEN):
        """
        Args:
            texts (List[str]): a list with texts to classify or to train the
                classifier on
            labels List[str]: a list with classification labels (optional)
            label_dict (dict): a dictionary mapping class names to class ids,
                to be passed to the validation data (optional)
            max_seq_length (int): maximal sequence length in tokens,
                texts will be stripped to this length
            model_name (str): transformer model name, needed to perform
                appropriate tokenization

        """

        self.captions = captions   
        self.labels = labels
        self.imagenet_words = imagenet_words
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.gan_model = gan_model
    
    def __len__(self):
        """
        Returns:
            int: length of the dataset
        """
        return len(self.captions)   

    def __getitem__(self, index):
        """Gets element of the dataset

        Args:
            index (int): index of the element in the dataset
        Returns:
            Single element by index
        """

        # encoding the text
        caption = self.captions[index]
        
        # class label
        label = self.labels[index]
        
        # get hidden states - input to the mapping model
        lm_hidden_states = encode_text_input(caption, 
                                             self.tokenizer,
                                             self.language_model,
                                             self.max_seq_length)
        
        # get BigGAN embeddings - target for the mapping model
        gan_embeddings = create_biggan_embeggings(caption,
                                                  label,
                                                  self.gan_model,
                                                  self.imagenet_words,
                                                  self.max_seq_length)
        
        return lm_hidden_states, gan_embeddings


# In[105]:


train_dataset = MappingDataset(train_captions, 
                               train_labels,
                               imagenet_words=all_words,
                               tokenizer=lm_tokenizer,
                               language_model=lm_model,
                               gan_model=gan_model)

toy_train_dataset = MappingDataset(train_captions[:100], 
                                   train_labels[:100],
                               imagenet_words=all_words,
                               tokenizer=lm_tokenizer,
                               language_model=lm_model,
                               gan_model=gan_model)


# In[75]:


val_dataset = MappingDataset(val_captions, 
                             val_labels,
                             imagenet_words=all_words,
                             tokenizer=lm_tokenizer,
                             language_model=lm_model,
                             gan_model=gan_model)


# In[76]:


len(train_dataset), len(val_dataset)


# In[77]:


train_captions[4]


# In[78]:


lm_hidden_states, gan_embeddings = train_dataset[4]


# In[79]:


lm_hidden_states.size(), gan_embeddings.size()


# ### Data loaders for the mapping model

# In[80]:


BATCH_SIZE = 1024


# In[106]:


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

toy_train_loader = DataLoader(dataset=toy_train_dataset,
                          batch_size=32, 
                          shuffle=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=BATCH_SIZE, 
                        shuffle=False)


# ### Mapping model

# In[110]:


BERT_HIDDEN_SIZE = 768
MAPPING_HIDDEN_SIZE = 256
MAPPING_OUTPUT_SIZE = 128

NUM_EPOCHS = 10
LEARN_RATE = 5e-3
loss_fn = torch.nn.MSELoss(reduction='mean')


# In[111]:


mapping_model = torch.nn.Sequential(
          torch.nn.Linear(BERT_HIDDEN_SIZE, MAPPING_OUTPUT_SIZE),
        )


# In[112]:


# mapping_model = torch.nn.Sequential(
#           torch.nn.Linear(BERT_HIDDEN_SIZE, MAPPING_HIDDEN_SIZE),
# #           torch.nn.Linear(BERT_HIDDEN_SIZE, MAPPING_OUTPUT_SIZE),
#           torch.nn.Dropout(0.5),
#           torch.nn.Linear(MAPPING_HIDDEN_SIZE, MAPPING_OUTPUT_SIZE),
#         )


# In[113]:


mapping_model


# ### Train the mapping model

# In[114]:


import catalyst
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback, CriterionCallback
from catalyst.dl.utils import set_global_seed, prepare_cudnn, plot_metrics


# In[131]:


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mapping_model.parameters(), lr=LEARN_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)


# In[132]:


runner = SupervisedRunner()


# In[127]:




os.environ['CUDA_VISIBLE_DEVICES'] = "1"    # can be changed in case of multiple GPUs onboard
set_global_seed(17)                         # reproducibility
prepare_cudnn(deterministic=True)           # reproducibility


# In[ ]:


LOG_DIR = '../data/train_logs_simple_model'


# In[55]:


#!rm -rf $LOG_DIR


# In[ ]:


os.environ['CUDA_VISIBLE_DEVICES'] = "1" 
runner.train(
    model=mapping_model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders={'train': train_loader, 'valid': val_loader},
    callbacks=[
        CriterionCallback(),
        EarlyStoppingCallback(patience=10)
    ],
    logdir=LOG_DIR,
    num_epochs=NUM_EPOCHS,
    verbose=False
)
