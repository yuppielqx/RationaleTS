 
import argparse
import os
import numpy as np
import torch
import warnings
import pickle as pkl
from transformers import AutoTokenizer,AutoModel

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='None')
parser.add_argument('--lm_model', type=str, default='bert')

args = parser.parse_args() 
city ='ny'
lm_model_name = 'bert'
lm_model = AutoModel.from_pretrained("./bert-base-cased", output_hidden_states=True)
if lm_model_name in ['deberta', 'bert', 'roberta', 'distilbert']:
    os.environ['TOKENIZERS_PARALLELISM'] = 'True'
    if args.lm_model == 'deberta':
        tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
    elif args.lm_model == 'bert':
        tokenizer = AutoTokenizer.from_pretrained("./bert-base-cased", output_hidden_states=True)
    elif args.lm_model == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    elif args.lm_model == 'distilbert':
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

data_path = os.path.join('dataset')
with open(os.path.join(data_path, 'indices.pkl'), 'rb') as f:
    indices = pkl.load(f)

with open(os.path.join(data_path, f'time_series_{city}.pkl'), 'rb') as f:
    time_series = torch.from_numpy(pkl.load(f))
    
with open(os.path.join(data_path, f'rain_{city}.pkl'), 'rb') as f:
    rain = torch.tensor(pkl.load(f))
    
texts = []

for i in indices:
    with open(os.path.join(data_path, 'weather_summary', f'{city}_{i}.txt'), 'r') as f:
        text = f.read()
        texts.append(text)

if lm_model_name in ['deberta', 'bert', 'roberta', 'distilbert']:
    texts = tokenizer(texts, padding=True, truncation=True, max_length=2048//1)
else:
    texts = np.array(texts)

data_size = len(indices)

num_train = int(data_size * 0.6)
num_test = int(data_size * 0.2)
num_vali = data_size - num_train - num_test
seq_len = 24
seq_len_day = seq_len // 24


idx_tr = np.arange(num_train - seq_len_day)
 
idx_va = np.arange(num_train - seq_len_day, num_train + num_vali - seq_len_day)
 
idx_te = np.arange(num_train + num_vali - seq_len_day, num_train + num_vali + num_test - seq_len_day)

idx_all = np.arange(num_train + num_vali + num_test - seq_len_day)

emb_texts = []
input_ids = []
att_masks = []
x_len = []
if lm_model_name in ['deberta', 'bert', 'roberta', 'distilbert']:
    for i in idx_tr:
        x_enc_text = {key: torch.tensor(val[i]) for key, val, in texts.items()} 
        input_ids.append(x_enc_text['input_ids'].unsqueeze(0))
        att_masks.append(x_enc_text['attention_mask'].unsqueeze(0))
    input_ids = torch.cat(input_ids,0)
    att_masks = torch.cat(att_masks,0) #torch.nested.nested_tensor(input_ids)#
        
    print('text embedding step start: training')    
    outputs = lm_model(input_ids=input_ids, attention_mask=att_masks,
                       output_hidden_states=True)['hidden_states'][-1].detach().numpy()
    
    np.save('input_ids_ny_tr.npy',input_ids.numpy())
    np.save('att_masks_ny_tr.npy',att_masks.numpy())
    np.save('text_emb_tr.npy', outputs)
    

    input_ids = []
    att_masks = []
    for i in idx_va:
        x_enc_text = {key: torch.tensor(val[i]) for key, val, in texts.items()} 
        input_ids.append(x_enc_text['input_ids'].unsqueeze(0))
        att_masks.append(x_enc_text['attention_mask'].unsqueeze(0))
    input_ids = torch.cat(input_ids,0)
    att_masks = torch.cat(att_masks,0) #torch.nested.nested_tensor(input_ids)#
        
    print('text embedding step start: validation')    
    outputs = lm_model(input_ids=input_ids, attention_mask=att_masks,
                       output_hidden_states=True)['hidden_states'][-1].detach().numpy()
    
    np.save('input_ids_ny_va.npy',input_ids.numpy())
    np.save('att_masks_ny_va.npy',att_masks.numpy())
    np.save('text_emb_va.npy', outputs)


    input_ids = []
    att_masks = []
    for i in idx_te:
        x_enc_text = {key: torch.tensor(val[i]) for key, val, in texts.items()} 
        input_ids.append(x_enc_text['input_ids'].unsqueeze(0))
        att_masks.append(x_enc_text['attention_mask'].unsqueeze(0))
    input_ids = torch.cat(input_ids,0)
    att_masks = torch.cat(att_masks,0) #torch.nested.nested_tensor(input_ids)#
        
    print('text embedding step start: testing')    
    outputs = lm_model(input_ids=input_ids, attention_mask=att_masks,
                       output_hidden_states=True)['hidden_states'][-1].detach().numpy()
    
    np.save('input_ids_ny_te.npy',input_ids.numpy())
    np.save('att_masks_ny_te.npy',att_masks.numpy())
    np.save('text_emb_te.npy', outputs)