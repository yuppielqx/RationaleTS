import torch
import numpy as np
import pickle as pkl
from sklearn.utils import shuffle
import os
from time import time
from model_weather_mm import cnn_pro_mm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from transformers import BertTokenizer
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
var = ['Humidity','Pressure','Temperature','Wind Speed','Wind Direction']

torch.set_default_dtype(torch.float32)
with open(os.path.join('dataset', 'indices.pkl'), 'rb') as f:
    indices = pkl.load(f)

filepath = 'dataset/time_series_ny.pkl'
with open(filepath, 'rb') as pklfile:
    ts = pkl.load(pklfile) # 45216 * 5
    
with open(os.path.join('dataset', 'rain_ny.pkl'), 'rb') as f:
    rain = np.array(pkl.load(f))
    
#text embedding
x_tr_text = np.load('text_emb_tr.npy') # N * L * D
x_va_text = np.load('text_emb_va.npy')
x_te_text = np.load('text_emb_te.npy')
#text token id
idx_tr_text = np.load('input_ids_ny_tr.npy')
idx_va_text = np.load('input_ids_ny_va.npy')
idx_te_text = np.load('input_ids_ny_te.npy') 

num_class = 2 #rain or not
data_size = len(indices)
num_train = int(data_size * 0.6)
num_test = int(data_size * 0.2)
num_vali = data_size - num_train - num_test

seq_len_day = 1
seq_len = 24
max_len_ts = seq_len
max_len_text = x_tr_text.shape[1]
pred_len = 1

train_idx = np.arange(num_train - seq_len_day)
val_idx = np.arange(num_train - seq_len_day, num_train + num_vali - seq_len_day)
test_idx = np.arange(num_train + num_vali - seq_len_day, num_train + num_vali + num_test - seq_len_day)

train_data = []
train_label = [] 
val_data = []
val_label = []
test_data = []
test_label = []

scaler = StandardScaler()
scaler.fit(ts[:indices[train_idx[-1]]+seq_len,:])
ts1 = scaler.transform(ts)

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)

for i in train_idx:
    idx = indices[i]
    train_data.append(ts1[idx:idx+seq_len,:])
    train_label.append(int(rain[i+pred_len]))
    
for i in val_idx:
    idx = indices[i]
    val_data.append(ts1[idx:idx+seq_len,:])
    val_label.append(int(rain[i+pred_len]))
    
for i in test_idx:
    idx = indices[i]
    test_data.append(ts1[idx:idx+seq_len,:])
    test_label.append(int(rain[i+pred_len]))

x_tr_ts, x_va_ts, x_te_ts = np.array(train_data), np.array(val_data), np.array(test_data)
y_tr, y_va, y_te = np.array(train_label), np.array(val_label), np.array(test_label) 

x_len_tr_ts = seq_len*np.ones(len(x_tr_ts)).astype(int)
x_len_va_ts = seq_len*np.ones(len(x_va_ts)).astype(int)
x_len_te_ts = seq_len*np.ones(len(x_te_ts)).astype(int)

x_len_tr_text = np.load('att_masks_ny_tr.npy').sum(1) # 计算没有被编码的token长度
x_len_va_text = np.load('att_masks_ny_va.npy').sum(1) 
x_len_te_text = np.load('att_masks_ny_te.npy').sum(1) 



dim_ts = 5
dim_text = 256 
kernel_sizes_ts = [4,8] 
kernel_sizes_text = [12]#[4]#[30,50] 
cin = 1
cout_ts = 32
cout_text = 64 
dropout_rate = 0.05 
k_ts = 20 
k_text = 10
lr = 0.0003
epochs = 40 
set_cuda = True
batchsz = 256 

lmd0 = 1.0
lmd1_ts = 0.25
lmd2_ts = 0.1
lmd3_ts = 0.05
lmd4_ts = 0.0 
dmin_ts = 3.0

lmd1_text = 0.15 
lmd2_text = 0.15
lmd3_text = 0.05
lmd4_text = 0.0 
dmin_text = 4.0

epoch_prottp = epochs

model = cnn_pro_mm(dim_ts=dim_ts,dim_text=dim_text,
                num_class=num_class,
                cin=cin,
                cout_ts=cout_ts,
                cout_text=cout_text,
                kernel_sizes_ts=kernel_sizes_ts,
                kernel_sizes_text=kernel_sizes_text,
                dropout_rate=dropout_rate,
                k_ts=k_ts,
                k_text=k_text,
                device=device)

if set_cuda is True:
    model.to(device)
 

print(model)

parameters = model.parameters()
opt = torch.optim.Adam(parameters, lr=lr)

# initialize prototypes
# ---
train_idx_track = np.arange(num_train - seq_len_day)
x_tr_ts, x_tr_text, y_tr, x_len_tr_ts, x_len_tr_text, idx_tr_text, train_idx_track = shuffle(x_tr_ts, x_tr_text, y_tr, x_len_tr_ts, x_len_tr_text, idx_tr_text,train_idx_track)
batch_x_ts = x_tr_ts[0:batchsz, :]
batch_x_text = x_tr_text[0:batchsz, :]
batch_y = y_tr[0:batchsz]#保证卷积只在有意义的token上执行
batch_msk_ts = [[[0 for _ in range(l - ks + 1)] + [float('inf')] * (max_len_ts - l) for l in x_len_tr_ts[0:batchsz]] for ks in kernel_sizes_ts]
batch_msk_text = [[[0 for _ in range(l - ks + 1)] + [float('inf')] * (max_len_text - l) for l in x_len_tr_text[0:batchsz]] for ks in kernel_sizes_text]


batch_x_ts = torch.tensor(batch_x_ts, dtype=torch.float32)
batch_x_text = torch.tensor(batch_x_text, dtype=torch.float32)
batch_y = torch.tensor(batch_y, dtype=torch.int64)
batch_msk_ts = [torch.tensor(mk, dtype=torch.float32) for mk in batch_msk_ts] 
batch_msk_text = [torch.tensor(mk, dtype=torch.float32) for mk in batch_msk_text]
if set_cuda is True:
    batch_x_ts = batch_x_ts.to(device)
    batch_x_text = batch_x_text.to(device)
    batch_y = batch_y.to(device)
    batch_msk_ts = [mk.to(device) for mk in batch_msk_ts]
    batch_msk_text = [mk.to(device) for mk in batch_msk_text]
 
model.init_prottp_ts(batch_x_ts, batch_y, batch_msk_ts, device)
model.init_prottp_text(batch_x_text, batch_y, batch_msk_text, device)

# training iteration
# --
for epoch in range(1, (epochs + 1)):
    t0 = time()

    # training
    # ---
    random_state = None
    if epoch == epochs:
        random_state = seed
    x_tr_ts, x_tr_text, y_tr, x_len_tr_ts, x_len_tr_text, idx_tr_text, train_idx_track = shuffle(x_tr_ts, x_tr_text, y_tr, x_len_tr_ts, x_len_tr_text, idx_tr_text,train_idx_track,random_state=random_state)

    model.train()
    cor_tr = 0
    ls_tr = 0
    ls_tr0 = 0
    ls_tr1 = 0
    ls_tr2 = 0
    ls_tr3 = 0
    ls_tr4 = 0
    
    ls_tr1_ts = 0
    ls_tr2_ts = 0
    ls_tr3_ts = 0
    ls_tr4_ts = 0
    
    ls_tr1_text = 0
    ls_tr2_text = 0
    ls_tr3_text= 0
    ls_tr4_text = 0
    
    cnt = 0

 
    if epoch == epochs:
        batch_x_tr_all_ts = torch.tensor(x_tr_ts, dtype=torch.float32)
        batch_x_tr_all_text = torch.tensor(x_tr_text, dtype=torch.float32)
        batch_x_tr_all_token = torch.tensor(idx_tr_text, dtype=torch.int64)
        batch_y_tr_all = torch.tensor(y_tr, dtype=torch.int64)
        batch_msk_tr_all_ts = [[[0 for _ in range(l - ks + 1)] + [float('inf')] * (max_len_ts - l) for l in x_len_tr_ts] for ks in kernel_sizes_ts] 
        batch_msk_tr_all_ts = [torch.tensor(mk, dtype=torch.float32) for mk in batch_msk_tr_all_ts]
        batch_msk_tr_all_text = [[[0 for _ in range(l - ks + 1)] + [float('inf')] * (max_len_text - l) for l in x_len_tr_text] for ks in kernel_sizes_text] 
        batch_msk_tr_all_text = [torch.tensor(mk, dtype=torch.float32) for mk in batch_msk_tr_all_text]
        if set_cuda is True:
            batch_x_tr_all_ts = batch_x_tr_all_ts.to(device)
            batch_x_tr_all_text = batch_x_tr_all_text.to(device)
            batch_x_tr_all_token = batch_x_tr_all_token.to(device)
            batch_y_tr_all = batch_y_tr_all.to(device)
            batch_msk_tr_all_ts = [mk.to(device) for mk in batch_msk_tr_all_ts]
            batch_msk_tr_all_text = [mk.to(device) for mk in batch_msk_tr_all_text]    

    for i in range(0, (len(y_tr) - batchsz), batchsz):
        if (len(y_tr) - i) < 2 * batchsz:
            batchsz_iter = len(y_tr) - i
        else:
            batchsz_iter = batchsz

        batch_x_ts = x_tr_ts[i:(i+batchsz_iter), :]
        batch_x_text = x_tr_text[i:(i+batchsz_iter), :]
        batch_y = y_tr[i:(i+batchsz_iter)]
        
        batch_x_token_idx = idx_tr_text[i:(i+batchsz_iter), :]
        
        batch_msk_ts = [[[0 for _ in range(l - ks + 1)] + [float('inf')] * (max_len_ts - l) for l in x_len_tr_ts[i:(i+batchsz_iter)]] for ks in kernel_sizes_ts]
        batch_msk_text = [[[0 for _ in range(l - ks + 1)] + [float('inf')] * (max_len_text - l) for l in x_len_tr_text[i:(i+batchsz_iter)]] for ks in kernel_sizes_text]
        
        
        batch_x_ts = torch.tensor(batch_x_ts, dtype=torch.float32)
        batch_x_text = torch.tensor(batch_x_text, dtype=torch.float32)
        batch_y = torch.tensor(batch_y, dtype=torch.int64)
        
        batch_x_token_idx = torch.tensor(batch_x_token_idx, dtype=torch.int64)
        
        batch_msk_ts = [torch.tensor(mk, dtype=torch.float32) for mk in batch_msk_ts] 
        batch_msk_text = [torch.tensor(mk, dtype=torch.float32) for mk in batch_msk_text]
        batch_y_one_hot = torch.zeros(batchsz_iter, num_class, dtype=torch.float32)
        batch_y_one_hot[torch.arange(batchsz_iter), batch_y] = 1.0

        if set_cuda is True:
            batch_x_ts = batch_x_ts.to(device)
            batch_x_text = batch_x_text.to(device)
            batch_x_token_idx = batch_x_token_idx.to(device)
            batch_y = batch_y.to(device)
            batch_y_one_hot = batch_y_one_hot.to(device)
            batch_msk_ts = [mk.to(device) for mk in batch_msk_ts]
            batch_msk_text = [mk.to(device) for mk in batch_msk_text]

        opt.zero_grad()
        logit, dists_ts, dists_text = model(batch_x_ts, batch_x_text, batch_msk_ts, batch_msk_text)
        
        ls, ls0, ls1_ts, ls2_ts, ls3_ts, ls4_ts = model.loss_ts(logit, batch_y_one_hot, dists_ts, lmd0,
                                                            lmd1_ts, lmd2_ts, lmd3_ts, lmd4_ts, dmin_ts)
        
        ls1, _, ls1_text, ls2_text, ls3_text, ls4_text = model.loss_text(logit, batch_y_one_hot, dists_text, lmd0,
                                                            lmd1_text, lmd2_text, lmd3_text, lmd4_text, dmin_text)
        
        ls += ls1
        
        ls.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=4)
        opt.step()
        model.fc.weight.data = model.fc.weight.clamp(0)

        ls_tr += ls.data
        ls_tr0 += ls0.data
        ls_tr1 += ls1_ts.data + ls1_text.data
        ls_tr2 += ls2_ts.data + ls2_text.data
        ls_tr3 += ls3_ts.data + ls3_text.data
        ls_tr4 += ls4_ts.data + ls4_text.data
        cor_tr += (torch.max(logit, 1)[1].data == batch_y.data).sum()
        cnt += 1


    ls_tr = ls_tr.cpu().numpy() / cnt
    ls_tr0 = ls_tr0.cpu().numpy() / cnt
    ls_tr1 = ls_tr1.cpu().numpy() / cnt
    ls_tr2 = ls_tr2.cpu().numpy() / cnt
    ls_tr3 = ls_tr3.cpu().numpy() / cnt
    ls_tr4 = ls_tr4.cpu().numpy() / cnt
    acc_tr = cor_tr.cpu().numpy() / len(y_tr)

    if epoch % epoch_prottp == 0:        
        p_seq_ts, p_subseq_idx_ts = model.projection_ts(batch_x_tr_all_ts, batch_y_tr_all, batch_msk_tr_all_ts)
        p_seq_text, p_subseq_idx_text, p_seq_token = model.projection_text(batch_x_tr_all_text, batch_y_tr_all, batch_msk_tr_all_text, batch_x_tr_all_token)


    batch_x_tr_ts = batch_x_ts
    batch_x_tr_text = batch_x_text
    batch_y_tr = batch_y
    batch_y_one_hot_tr = batch_y_one_hot
    batch_msk_tr_ts = batch_msk_ts
    batch_msk_tr_text = batch_msk_text
    

    # validation
    # ---
    x_va_ts, x_va_text, y_va, x_len_va_ts, x_len_va_text = shuffle(x_va_ts, x_va_text, y_va, x_len_va_ts, x_len_va_text)
    model.eval()
    cor_va = 0
    ls_va = 0
    ls_va0 = 0
    ls_va1 = 0
    ls_va2 = 0
    ls_va3 = 0
    ls_va4 = 0
    cnt = 0
    
    ls_va1_ts = 0
    ls_va2_ts = 0
    ls_va3_ts = 0
    ls_va4_ts = 0
    
    ls_va1_text = 0
    ls_va2_text = 0
    ls_va3_text= 0
    ls_va4_text = 0
    
    pred_val = []
    gt_val = []
    logits_val = []

    if len(y_va) > batchsz:
        va_iter = range(0, (len(y_va) - batchsz), batchsz)
    else:
        va_iter = [0]

    for i in va_iter:
        if (len(y_va) - i) < 2 * batchsz:
            batchsz_iter = len(y_va) - i
        else:
            batchsz_iter = batchsz

        batch_x_ts = x_va_ts[i:(i+batchsz_iter), :]
        batch_x_text = x_va_text[i:(i+batchsz_iter), :]
        batch_y = y_va[i:(i+batchsz_iter)]
        batch_msk_ts = [[[0 for _ in range(l - ks + 1)] + [float('inf')] * (max_len_ts - l) for l in x_len_va_ts[i:(i+batchsz_iter)]] for ks in kernel_sizes_ts]
        batch_msk_text = [[[0 for _ in range(l - ks + 1)] + [float('inf')] * (max_len_text - l) for l in x_len_va_text[i:(i+batchsz_iter)]] for ks in kernel_sizes_text]
        
        batch_x_ts = torch.tensor(batch_x_ts, dtype=torch.float32)
        batch_x_text = torch.tensor(batch_x_text, dtype=torch.float32)
        batch_y = torch.tensor(batch_y, dtype=torch.int64)
        batch_msk_ts = [torch.tensor(mk, dtype=torch.float32) for mk in batch_msk_ts]  # num_kernels * [(batch, len - kernel_size + 1)]
        batch_msk_text = [torch.tensor(mk, dtype=torch.float32) for mk in batch_msk_text]
        batch_y_one_hot = torch.zeros(batchsz_iter, num_class, dtype=torch.float32)
        batch_y_one_hot[torch.arange(batchsz_iter), batch_y] = 1.0

        if set_cuda is True:
            batch_x_ts = batch_x_ts.to(device)
            batch_x_text = batch_x_text.to(device)
            batch_y = batch_y.to(device)
            batch_y_one_hot = batch_y_one_hot.to(device)
            batch_msk_ts = [mk.to(device) for mk in batch_msk_ts]
            batch_msk_text = [mk.to(device) for mk in batch_msk_text]

        logit, dists_ts, dists_text = model(batch_x_ts, batch_x_text, batch_msk_ts, batch_msk_text)
        
        ls, ls0, ls1_ts, ls2_ts, ls3_ts, ls4_ts = model.loss_ts(logit, batch_y_one_hot, dists_ts, lmd0,
                                                            lmd1_ts, lmd2_ts, lmd3_ts, lmd4_ts, dmin_ts)
        
        ls1, _, ls1_text, ls2_text, ls3_text, ls4_text = model.loss_text(logit, batch_y_one_hot, dists_text, lmd0,
                                                            lmd1_text, lmd2_text, lmd3_text, lmd4_text, dmin_text)
        
        ls += ls1

        ls_va += ls.data
        ls_va0 += ls0.data
        ls_va1 += ls1_ts.data + ls1_text.data
        ls_va2 += ls2_ts.data + ls2_text.data
        ls_va3 += ls3_ts.data + ls3_text.data
        ls_va4 += ls4_ts.data + ls4_text.data
        cor_va += (torch.max(logit, 1)[1].data == batch_y.data).sum()
        cnt += 1
        
        pred_val.append(torch.max(logit, 1)[1].data)
        gt_val.append(batch_y.data)
        logits_val.append(logit.data)

    ls_va = ls_va.cpu().numpy() / cnt
    ls_va0 = ls_va0.cpu().numpy() / cnt
    ls_va1 = ls_va1.cpu().numpy() / cnt
    ls_va2 = ls_va2.cpu().numpy() / cnt
    ls_va3 = ls_va3.cpu().numpy() / cnt
    ls_va4 = ls_va4.cpu().numpy() / cnt
    acc_va = cor_va.cpu().numpy() / len(y_va)
    
    
    pred_val = torch.cat(pred_val,0).cpu().numpy()
    gt_val = torch.cat(gt_val,0).cpu().numpy()
    logits_val = torch.cat(logits_val,0).cpu().numpy()

    gt_val_one_hot = np.zeros((len(gt_val), num_class))
    gt_val_one_hot[np.arange(len(gt_val)), gt_val] = 1.0
    
    f1_mi_va = f1_score(np.array(gt_val), np.array(pred_val), average='micro')
    f1_ma_va = f1_score(np.array(gt_val), np.array(pred_val), average='macro')
    auc_va = roc_auc_score(gt_val_one_hot,logits_val)

    print('training: epoch={:d}'.format(epoch),
          'loss={:.3f}'.format(ls_tr),
          'loss0={:.3f}'.format(ls_tr0),
          'loss1={:.3f}'.format(ls_tr1),
          'loss2={:.3f}'.format(ls_tr2),
          'loss3={:.3f}'.format(ls_tr3),
          'loss4={:.3f}'.format(ls_tr4),
          '| validation: loss={:.3f}'.format(ls_va),
          'loss0={:.3f}'.format(ls_va0),
          'loss1={:.3f}'.format(ls_va1),
          'loss2={:.3f}'.format(ls_va2),
          'loss3={:.3f}'.format(ls_va3),
          'loss4={:.3f}'.format(ls_va4),
          'F1-micro={:.4f}'.format(f1_mi_va),
          'F1-macro={:.4f}'.format(f1_ma_va),
          'AUC={:.4f}'.format(auc_va),
          )

# testing

cor_te = 0
ls_te = 0
cnt = 0

input_test_ts = []
input_test_text = []
pred_test = []
gt_test = []
logits_test = []
dists_test_ts = []
dists_test_text = []
if len(y_te) > batchsz:
    te_iter = range(0, (len(y_te) - batchsz), batchsz)
else:
    te_iter = [0]

for i in te_iter:
    if (len(y_te) - i) < 2 * batchsz:
        batchsz_iter = len(y_te) - i
    else:
        batchsz_iter = batchsz

    batch_x_ts = x_te_ts[i:(i + batchsz_iter), :]
    batch_x_text = x_te_text[i:(i + batchsz_iter), :]
    batch_x_token_idx = idx_te_text[i:(i+batchsz_iter), :]
    batch_y = y_te[i:(i + batchsz_iter)]
    batch_x_len_ts = x_len_te_ts[i:(i+batchsz_iter)]
    batch_msk_ts = [[[0 for _ in range(l - ks + 1)] + [float('inf')] * (max_len_ts - l) for l in batch_x_len_ts] for ks in kernel_sizes_ts]
    batch_x_len_text = x_len_te_text[i:(i+batchsz_iter)]
    batch_msk_text = [[[0 for _ in range(l - ks + 1)] + [float('inf')] * (max_len_text - l) for l in batch_x_len_text] for ks in kernel_sizes_text]
    
    
    
    batch_x_ts = torch.tensor(batch_x_ts, dtype=torch.float32)
    batch_x_text = torch.tensor(batch_x_text, dtype=torch.float32)
    batch_x_token_idx = torch.tensor(batch_x_token_idx, dtype=torch.int64)
    batch_y = torch.tensor(batch_y, dtype=torch.int64)
    batch_msk_ts = [torch.tensor(mk, dtype=torch.float32) for mk in batch_msk_ts]  # num_kernels * [(batch, len - kernel_size + 1)]
    batch_msk_text = [torch.tensor(mk, dtype=torch.float32) for mk in batch_msk_text]
    batch_y_one_hot = torch.zeros(batchsz_iter, num_class, dtype=torch.float32)
    batch_y_one_hot[torch.arange(batchsz_iter), batch_y] = 1.0

    if set_cuda is True:
        batch_x_ts = batch_x_ts.to(device)
        batch_x_text = batch_x_text.to(device)
        batch_x_token_idx = batch_x_token_idx.to(device)
        batch_y = batch_y.to(device)
        batch_y_one_hot = batch_y_one_hot.to(device)
        batch_msk_ts = [mk.to(device) for mk in batch_msk_ts]
        batch_msk_text = [mk.to(device) for mk in batch_msk_text]

    logit, dists_ts, dists_text = model(batch_x_ts, batch_x_text, batch_msk_ts, batch_msk_text)
     
    ls, ls0, ls1_ts, ls2_ts, ls3_ts, ls4_ts = model.loss_ts(logit, batch_y_one_hot, dists_ts, lmd0,
                                                        lmd1_ts, lmd2_ts, lmd3_ts, lmd4_ts, dmin_ts)
    
    ls1, _, ls1_text, ls2_text, ls3_text, ls4_text = model.loss_text(logit, batch_y_one_hot, dists_text, lmd0,
                                                        lmd1_text, lmd2_text, lmd3_text, lmd4_text, dmin_text)
    
    ls += ls1

    ls_te += ls.data
    cor_te += (torch.max(logit, 1)[1].data == batch_y.data).sum()
    cnt += 1
    
    input_test_ts.append(batch_x_ts.data) 
    input_test_text.append(batch_x_token_idx.data)
    pred_test.append(torch.max(logit, 1)[1].data)
    gt_test.append(batch_y.data)
    logits_test.append(logit.data)
    dists_test_ts.append(dists_ts) 
    dists_test_text.append(dists_text)

input_test_ts = torch.cat(input_test_ts,0).cpu().numpy()
input_test_text = torch.cat(input_test_text,0).cpu().numpy()
pred_test = torch.cat(pred_test,0).cpu().numpy()
gt_test = torch.cat(gt_test,0).cpu().numpy()
logits_test = torch.cat(logits_test,0).cpu().numpy()

gt_test_one_hot = np.zeros((len(gt_test), num_class))
gt_test_one_hot[np.arange(len(gt_test)), gt_test] = 1.0


ls_te = ls_te.cpu().numpy() / cnt
acc_te = cor_te.cpu().numpy() / len(y_te)

f1_mi = f1_score(np.array(gt_test), np.array(pred_test), average='micro')

f1_ma = f1_score(np.array(gt_test), np.array(pred_test), average='macro')


auc = roc_auc_score(gt_test_one_hot,logits_test)


print('testing: loss={:.3f}'.format(ls_te),
      'micro-F1={:.4f}'.format(f1_mi),
      'macro-F1={:.4f}'.format(f1_ma),
      'auc={:.4f}'.format(auc))



##find case-based explanations
def find_k_min_values_and_indices(array, k):
    """
    Finds the minimum k values and their indices in a 3D array.

    Parameters:
    array (np.ndarray): The 3D array to search.
    k (int): The number of minimum values to find.

    Returns:
    min_values (np.ndarray): The minimum k values.
    min_indices (np.ndarray): The indices of the minimum k values.
    """
    # Flatten the array
    flat_array = array.flatten()
    
    # Find the indices of the minimum k values
    flat_indices = np.argpartition(flat_array, k)[:k]
    
    # Get the minimum k values
    min_values = flat_array[flat_indices]
    
    # Sort the k minimum values and their indices
    sorted_order = np.argsort(min_values)
    sorted_flat_indices = flat_indices[sorted_order]
    sorted_min_values = min_values[sorted_order]
    
    # Convert flattened indices back to 3D indices
    min_indices = np.array(np.unravel_index(sorted_flat_indices, array.shape))
    
    return sorted_min_values, min_indices


##load the testing data 
with open(os.path.join('dataset', 'indices.pkl'), 'rb') as f:
    indices = pkl.load(f)

filepath = 'dataset/time_series_ny.pkl'
with open(filepath, 'rb') as pklfile:
    ts = pkl.load(pklfile)
    
with open(os.path.join('dataset', 'rain_ny.pkl'), 'rb') as f:
    rain = np.array(pkl.load(f))
    
#text embedding
x_tr_text = np.load('text_emb_tr.npy')
x_va_text = np.load('text_emb_va.npy')
x_te_text = np.load('text_emb_te.npy')
#text token id
idx_tr_text = np.load('input_ids_ny_tr.npy')
idx_va_text = np.load('input_ids_ny_va.npy')
idx_te_text = np.load('input_ids_ny_te.npy') 

num_class = 2
data_size = len(indices)
num_train = int(data_size * 0.6)
num_test = int(data_size * 0.2)
num_vali = data_size - num_train - num_test

seq_len_day = 1
seq_len = 24
max_len_ts = seq_len
max_len_text = x_tr_text.shape[1]
pred_len = 1

train_idx = np.arange(num_train - seq_len_day)
val_idx = np.arange(num_train - seq_len_day, num_train + num_vali - seq_len_day)
test_idx = np.arange(num_train + num_vali - seq_len_day, num_train + num_vali + num_test - seq_len_day)

train_data = []
train_label = [] 
val_data = []
val_label = []
test_data = []
test_label = []

scaler = StandardScaler()
scaler.fit(ts[:indices[train_idx[-1]]+seq_len,:])
ts1 = scaler.transform(ts)


for i in train_idx:
    idx = indices[i]
    train_data.append(ts1[idx:idx+seq_len,:])
    train_label.append(int(rain[i+pred_len]))
    
for i in val_idx:
    idx = indices[i]
    val_data.append(ts1[idx:idx+seq_len,:])
    val_label.append(int(rain[i+pred_len]))
    
for i in test_idx:
    idx = indices[i]
    test_data.append(ts1[idx:idx+seq_len,:])
    test_label.append(int(rain[i+pred_len]))

x_tr_ts, x_va_ts, x_te_ts = np.array(train_data), np.array(val_data), np.array(test_data)
y_tr, y_va, y_te = np.array(train_label), np.array(val_label), np.array(test_label) 

x_len_tr_ts = seq_len*np.ones(len(x_tr_ts)).astype(int)
x_len_va_ts = seq_len*np.ones(len(x_va_ts)).astype(int)
x_len_te_ts = seq_len*np.ones(len(x_te_ts)).astype(int)

x_len_tr_text = np.load('att_masks_ny_tr.npy').sum(1) 
x_len_va_text = np.load('att_masks_ny_va.npy').sum(1)
x_len_te_text = np.load('att_masks_ny_te.npy').sum(1)

batch_x_tr_all_ts = torch.tensor(x_tr_ts, dtype=torch.float32)
batch_x_tr_all_text = torch.tensor(x_tr_text, dtype=torch.float32)
batch_x_tr_all_token = torch.tensor(idx_tr_text, dtype=torch.int64)
batch_y_tr_all = torch.tensor(y_tr, dtype=torch.int64)
batch_msk_tr_all_ts = [[[0 for _ in range(l - ks + 1)] + [float('inf')] * (max_len_ts - l) for l in x_len_tr_ts] for ks in kernel_sizes_ts] 
batch_msk_tr_all_ts = [torch.tensor(mk, dtype=torch.float32) for mk in batch_msk_tr_all_ts]
batch_msk_tr_all_text = [[[0 for _ in range(l - ks + 1)] + [float('inf')] * (max_len_text - l) for l in x_len_tr_text] for ks in kernel_sizes_text] 
batch_msk_tr_all_text = [torch.tensor(mk, dtype=torch.float32) for mk in batch_msk_tr_all_text]
if set_cuda is True:
    batch_x_tr_all_ts = batch_x_tr_all_ts.to(device)
    batch_x_tr_all_text = batch_x_tr_all_text.to(device)
    batch_x_tr_all_token = batch_x_tr_all_token.to(device)
    batch_y_tr_all = batch_y_tr_all.to(device)
    batch_msk_tr_all_ts = [mk.to(device) for mk in batch_msk_tr_all_ts]
    batch_msk_tr_all_text = [mk.to(device) for mk in batch_msk_tr_all_text]

logit_tr_all, dists_ts_tr_all, dists_text_tr_all = model(batch_x_tr_all_ts, batch_x_tr_all_text, batch_msk_tr_all_ts, batch_msk_tr_all_text)
pred_tr_all = torch.max(logit_tr_all, 1)[1].data.cpu().numpy()
gt_tr_all = batch_y_tr_all.cpu().numpy()

train_expl = {}
topk=20
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
for instance_idx in range(len(train_idx)):
    _instance_idx = train_idx[instance_idx]
    input_text_token = batch_x_tr_all_token[instance_idx]

    train_expl[int(_instance_idx)] = []
    
    for i in range(len(dists_text_tr_all)): 
  
        dists = dists_text_tr_all 
        array = dists[i].detach().cpu().numpy()[instance_idx] 
        min_vals, min_indices = find_k_min_values_and_indices(array, topk) 
        seg_idx, ptp_class_idx, ptp_k_idx = min_indices 
        ptp_length = kernel_sizes_text[i]

        selected_id, ori_id = np.unique(np.vstack((ptp_class_idx, ptp_k_idx)).T,axis=0,return_index=True)

        selected_id = selected_id[np.argsort(ori_id)]
        ori_id  =sorted(ori_id)
        topk_selected = len(selected_id)

        for j in range(topk_selected):
 
            ori_txt_seq = p_seq_token[i][selected_id[j,0]*k_text + selected_id[j,1]]
            s_e = p_subseq_idx_text[i][selected_id[j,0]*k_text + selected_id[j,1]]

            train_expl[int(_instance_idx)].append({'Class': int(selected_id[j,0]) , 'Similarity': float(np.exp(-min_vals[ori_id[j]])) , 'Prototype': tokenizer.decode(ori_txt_seq[s_e]), 
            'Input Segment': tokenizer.decode(input_text_token[seg_idx[ori_id[j]]:seg_idx[ori_id[j]]+kernel_sizes_text[i]])})
        

with open('./expl_results/train_expl.json', 'w') as json_file:
    json.dump(train_expl, json_file)

np.save('./expl_results/gt_train.npy',gt_tr_all)
np.save('./expl_results/pred_train.npy',pred_tr_all)
np.save('./expl_results/logit_train.npy',logit_tr_all.detach().cpu().numpy())



batch_x_va_all_ts = torch.tensor(x_va_ts, dtype=torch.float32)
batch_x_va_all_text = torch.tensor(x_va_text, dtype=torch.float32)
batch_x_va_all_token = torch.tensor(idx_va_text, dtype=torch.int64)
batch_y_va_all = torch.tensor(y_va, dtype=torch.int64)
batch_msk_va_all_ts = [[[0 for _ in range(l - ks + 1)] + [float('inf')] * (max_len_ts - l) for l in x_len_va_ts] for ks in kernel_sizes_ts] 
batch_msk_va_all_ts = [torch.tensor(mk, dtype=torch.float32) for mk in batch_msk_va_all_ts]
batch_msk_va_all_text = [[[0 for _ in range(l - ks + 1)] + [float('inf')] * (max_len_text - l) for l in x_len_va_text] for ks in kernel_sizes_text] 
batch_msk_va_all_text = [torch.tensor(mk, dtype=torch.float32) for mk in batch_msk_va_all_text]
if set_cuda is True:
    batch_x_va_all_ts = batch_x_va_all_ts.to(device)
    batch_x_va_all_text = batch_x_va_all_text.to(device)
    batch_x_va_all_token = batch_x_va_all_token.to(device)
    batch_y_va_all = batch_y_va_all.to(device)
    batch_msk_va_all_ts = [mk.to(device) for mk in batch_msk_va_all_ts]
    batch_msk_va_all_text = [mk.to(device) for mk in batch_msk_va_all_text]

logit_va_all, dists_ts_va_all, dists_text_va_all = model(batch_x_va_all_ts, batch_x_va_all_text, batch_msk_va_all_ts, batch_msk_va_all_text)
pred_va_all = torch.max(logit_va_all, 1)[1].data.cpu().numpy()
gt_va_all = batch_y_va_all.cpu().numpy()


val_expl = {}
topk=20
#tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
for instance_idx in range(len(val_idx)):
    _instance_idx = val_idx[instance_idx]
    input_text_token = batch_x_va_all_token[instance_idx]

    val_expl[int(_instance_idx)] = []
    
    for i in range(len(dists_text_va_all)): 
  
        dists = dists_text_va_all 
        array = dists[i].detach().cpu().numpy()[instance_idx] 
        min_vals, min_indices = find_k_min_values_and_indices(array, topk) 
        seg_idx, ptp_class_idx, ptp_k_idx = min_indices 
        ptp_length = kernel_sizes_text[i]

        selected_id, ori_id = np.unique(np.vstack((ptp_class_idx, ptp_k_idx)).T,axis=0,return_index=True)

        selected_id = selected_id[np.argsort(ori_id)]
        ori_id = sorted(ori_id)
        topk_selected = len(selected_id)

        for j in range(topk_selected):
 
            ori_txt_seq = p_seq_token[i][selected_id[j,0]*k_text + selected_id[j,1]]
            s_e = p_subseq_idx_text[i][selected_id[j,0]*k_text + selected_id[j,1]]

            val_expl[int(_instance_idx)].append({'Class': int(selected_id[j,0]) , 'Similarity': float(np.exp(-min_vals[ori_id[j]])) , 'Prototype': tokenizer.decode(ori_txt_seq[s_e]), 
            'Input Segment': tokenizer.decode(input_text_token[seg_idx[ori_id[j]]:seg_idx[ori_id[j]]+kernel_sizes_text[i]])})
        

with open('./expl_results/val_expl.json', 'w') as json_file:
    json.dump(val_expl, json_file)

np.save('./expl_results/gt_val.npy',gt_va_all)
np.save('./expl_results/pred_val.npy',pred_va_all)
np.save('./expl_results/logit_val.npy',logit_va_all.detach().cpu().numpy())


batch_x_te_all_ts = torch.tensor(x_te_ts, dtype=torch.float32)
batch_x_te_all_text = torch.tensor(x_te_text, dtype=torch.float32)
batch_x_te_all_token = torch.tensor(idx_te_text, dtype=torch.int64)
batch_y_te_all = torch.tensor(y_te, dtype=torch.int64)
batch_msk_te_all_ts = [[[0 for _ in range(l - ks + 1)] + [float('inf')] * (max_len_ts - l) for l in x_len_te_ts] for ks in kernel_sizes_ts] 
batch_msk_te_all_ts = [torch.tensor(mk, dtype=torch.float32) for mk in batch_msk_te_all_ts]
batch_msk_te_all_text = [[[0 for _ in range(l - ks + 1)] + [float('inf')] * (max_len_text - l) for l in x_len_te_text] for ks in kernel_sizes_text] 
batch_msk_te_all_text = [torch.tensor(mk, dtype=torch.float32) for mk in batch_msk_te_all_text]
if set_cuda is True:
    batch_x_te_all_ts = batch_x_te_all_ts.to(device)
    batch_x_te_all_text = batch_x_te_all_text.to(device)
    batch_x_te_all_token = batch_x_te_all_token.to(device)
    batch_y_te_all = batch_y_te_all.to(device)
    batch_msk_te_all_ts = [mk.to(device) for mk in batch_msk_te_all_ts]
    batch_msk_te_all_text = [mk.to(device) for mk in batch_msk_te_all_text]

logit_te_all, dists_ts_te_all, dists_text_te_all = model(batch_x_te_all_ts, batch_x_te_all_text, batch_msk_te_all_ts, batch_msk_te_all_text)
pred_te_all = torch.max(logit_te_all, 1)[1].data.cpu().numpy()
gt_te_all = batch_y_te_all.cpu().numpy()

#save testing explanations
test_expl = {}
topk=20
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
for instance_idx in range(len(test_idx)):
    _instance_idx = test_idx[instance_idx]
    input_text_token = batch_x_te_all_token[instance_idx]

    test_expl[int(_instance_idx)] = []
    
    for i in range(len(dists_text_te_all)): 
  
        dists = dists_text_te_all 
        array = dists[i].detach().cpu().numpy()[instance_idx] 
        min_vals, min_indices = find_k_min_values_and_indices(array, topk) 
        seg_idx, ptp_class_idx, ptp_k_idx = min_indices 
        ptp_length = kernel_sizes_text[i]

        selected_id, ori_id = np.unique(np.vstack((ptp_class_idx, ptp_k_idx)).T,axis=0,return_index=True)

        selected_id = selected_id[np.argsort(ori_id)]
        ori_id  =sorted(ori_id)
        topk_selected = len(selected_id)

        for j in range(topk_selected):
 
            ori_txt_seq = p_seq_token[i][selected_id[j,0]*k_text + selected_id[j,1]]
            s_e = p_subseq_idx_text[i][selected_id[j,0]*k_text + selected_id[j,1]]

            test_expl[int(_instance_idx)].append({'Class': int(selected_id[j,0]) , 'Similarity': float(np.exp(-min_vals[ori_id[j]])) , 'Prototype': tokenizer.decode(ori_txt_seq[s_e]), 
            'Input Segment': tokenizer.decode(input_text_token[seg_idx[ori_id[j]]:seg_idx[ori_id[j]]+kernel_sizes_text[i]])})
        

with open('./expl_results/test_expl.json', 'w') as json_file:
    json.dump(test_expl, json_file)

np.save('./expl_results/gt_test.npy',gt_te_all)
np.save('./expl_results/pred_test.npy',pred_te_all)
np.save('./expl_results/logit_test.npy',logit_te_all.detach().cpu().numpy())

