import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import re
import json
import glob
from copy import deepcopy
from collections import defaultdict
from functools import partial
from imblearn.under_sampling import RandomUnderSampler
import argparse
import pandas as pd
import numpy as np
from nltk import sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import unidecode
from tqdm.notebook import tqdm
import string
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torchcrf import CRF
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint', type=str, default='distilbert-base-uncased', help='model checkpoint')
parser.add_argument('--max_len', type=int, default=256, help='Maximum of tokens in a sentence')
args = parser.parse_args()

device='cuda' if torch.cuda.is_available() else 'cpu'
model_checkpoint=args.model_checkpoint
tokenizer=AutoTokenizer.from_pretrained(model_checkpoint)

def clean_text(txt):
    return [re.sub('[^A-Za-z0-9]+', ' ', str(t).lower()) for t in txt]
torch.manual_seed(1)

max_len=args.max_len

# Load data
train_df=pd.read_csv('data/processed_data/processed_train_df.csv')
labels_list=[i.split('|') for i in train_df.label.dropna().unique()]
labels_list=np.unique([i for label in labels_list for i in label])
input_path='data/raw_data/'
basepath=input_path+'coleridgeinitiative-show-us-the-data/'
train_df=pd.read_csv(basepath+'train.csv')
sample_sub = pd.read_csv(basepath+'sample_submission.csv')

# Process test_data
train_files_path=basepath+'train/'
test_files_path=basepath+'test/'
def read_append_return(filename, train_files_path=train_files_path, output='text'):
    json_path = os.path.join(train_files_path, (filename+'.json'))
    headings = []
    contents = []
    combined = []
    with open(json_path, 'r') as f:
        json_decode = json.load(f)
        for data in json_decode:
            headings.extend(sent_tokenize(unidecode.unidecode(data.get('section_title'))))
            contents.extend(sent_tokenize(unidecode.unidecode(data.get('text'))))
            combined.extend(sent_tokenize(unidecode.unidecode(data.get('section_title'))))
            combined.extend(sent_tokenize(unidecode.unidecode(data.get('text'))))
    
    if output == 'text':
        return contents
    elif output == 'head':
        return headings
    else:
        return combined
    
tqdm.pandas()
sample_sub['text'] = sample_sub['Id'].progress_apply(partial(read_append_return, train_files_path=test_files_path))

sample_sub['cleaned_text']=sample_sub.text.progress_apply(clean_text)

test_texts=[[t.strip() for t in txt] for txt in sample_sub.cleaned_text]

sub_labels=[]
for text in tqdm(sample_sub.cleaned_text):
    text=' '.join(text)
    tmp=sorted([label.strip() for label in labels_list if label in text],key=lambda x: len(x))
    sub_labels.append(tmp)

test_df=pd.DataFrame(zip(test_texts,sample_sub.Id),columns=['text','Id'])

# Define model
START_TAG=tokenizer.cls_token
STOP_TAG=tokenizer.sep_token
PAD_TAG=tokenizer.pad_token

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4, PAD_TAG:-1}

class my_model(nn.Module):
    def __init__(self,backbone,tag_to_ix):
        super(my_model,self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        # feature extraction
        self.backbone=backbone
        self.hidden_dim=backbone(**tokenizer('test',return_tensors='pt'))[0].shape[-1]
        # Maps the output of the backbone into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        self.aux_fc = nn.Linear(self.hidden_dim,1)
        # CRF
        self.crf = CRF(self.tagset_size, batch_first=True)
        # loss func
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs, labels=None, cls_labels=None):
        # Get the emission scores from the backbone
        outputs = self.backbone(**inputs).last_hidden_state
        emission = self.hidden2tag(outputs)
        cls_output = self.aux_fc(outputs[:,0,:])
        
        # Return result
        if labels is not None and cls_labels is not None:
            crf_loss = -self.crf(nn.functional.log_softmax(emission,2), labels, mask=inputs['attention_mask'].bool(), reduction='mean')
            cls_loss = self.loss_fn(cls_output,cls_labels)
            loss = crf_loss+cls_loss
            return loss
        else:
            prediction = self.crf.decode(emission,mask=inputs['attention_mask'].bool())
            return prediction

def gen_label(text,label):
    encoded_text=[tokenizer.cls_token] + tokenizer.tokenize(text) + [tokenizer.sep_token]
    result=[tokenizer.cls_token] + ['O']*len(tokenizer.tokenize(text)) + [tokenizer.sep_token]
    for label in label:
        if label=='':
            continue
        encoded_label=tokenizer.tokenize(label)
        for i,token in enumerate(encoded_text):
            if token==encoded_label[0] and encoded_text[i:i+len(encoded_label)]==encoded_label:
                result[i]='B'
                result[i+1:i+len(encoded_label)]=['I']*(len(encoded_label)-1)
    return [tag_to_ix[i] for i in result]

def gen_label_batch(texts,labels):
    tags=[gen_label(*inputs)[:max_len] for inputs in zip(texts,labels)]
    max_length=max([len(tag) for tag in tags])
    if tokenizer.padding_side=='right':
        return torch.tensor([tag+[tag_to_ix[PAD_TAG]]*(max_length-len(tag)) for tag in tags], dtype=torch.long, device=device).view(len(texts),-1)
    else:
        return torch.tensor([[tag_to_ix[PAD_TAG]]*(max_length-len(tag))+tag for tag in tags], dtype=torch.long, device=device).view(len(texts),-1)

# create model
backbone=AutoModel.from_pretrained(model_checkpoint)
model=my_model(backbone,tag_to_ix).to(device)
model.load_state_dict(torch.load(f'model_checkpoint/{model_checkpoint}.bin'))

# Helper function
model.eval()
def get_result(inputs,outputs):
    try:
        outputs_txt=''.join([str(i) for i in outputs])
        groups=[m.span() for m in re.finditer(r'[01]+',outputs_txt)]
        prediction=[]
        for i,j in groups:
            while tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][i].item()).startswith('##'):
                i-=1
            while tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][j].item()).startswith('##'):
                j+=1
            prediction.append(tokenizer.decode(inputs['input_ids'][0][i:j]))
        return prediction
    except:
        return []

def filter_labels(labels):
    tmp=[]
    labels=np.unique(labels)
    for index,label in enumerate(labels):
        try:
            if sum([label in ref for ref in labels])>1:
                continue
            else:
                tmp.append(str(label).strip())
        except:
            tmp.append(str(label).strip())
    return '|'.join(sorted(tmp))

sigmoid=nn.Sigmoid()

def get_cls_label(inputs):
    outputs = model.backbone(**inputs).last_hidden_state
    cls_output = model.aux_fc(outputs[:,0,:])
    return sigmoid(cls_output)

# get prediction
predictions=[]
for i,txt in enumerate(tqdm(test_df.text)):
    tmp_pred=[]
    for text in txt:
        inputs=tokenizer(text,return_tensors='pt',padding=True,truncation=True,max_length=max_len)
        inputs={k:v.to(device) for k,v in inputs.items()}
        if get_cls_label(inputs)[0].item() < 0.5:
            continue
        outputs=np.array(model(inputs)[0])
        tmp_pred.extend(get_result(inputs,outputs))
    predictions.append(filter_labels(sorted(clean_text(tmp_pred+sub_labels[i]))))

test_df['PredictionString']=predictions
submission=test_df.drop('text',axis=1)
submission.to_csv('data/processed_data/submission.csv',index=False)