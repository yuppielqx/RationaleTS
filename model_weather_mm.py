import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 

def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    init = (2 * init_range) * torch.rand(shape[0], shape[1]) - init_range
    return init

class cnn_pro_mm(nn.Module):
    def __init__(self, dim_ts, dim_text, num_class, cin, cout_ts, cout_text, kernel_sizes_ts, kernel_sizes_text, dropout_rate, k_ts, k_text, device='cpu'):
        super(cnn_pro_mm, self).__init__()
        self.num_kernels_ts = len(kernel_sizes_ts)
        self.num_kernels_text = len(kernel_sizes_text)
        self.num_class = num_class
        self.k_ts = k_ts
        self.k_text = k_text
        self.convs_ts = nn.ModuleList(nn.Conv2d(in_channels=cin,
                                             out_channels=cout_ts,
                                             kernel_size=(kernel_sizes_ts[i], dim_ts),
                                             stride=1,
                                             padding=0,
                                             dilation=1,
                                             groups=1,
                                             bias=False) for i in range(self.num_kernels_ts))
        
        self.text_emb_fc = nn.Linear(768, dim_text)
        self.text_dropout = nn.Dropout(0.1)
        self.convs_text = nn.ModuleList(nn.Conv2d(in_channels=cin,
                                             out_channels=cout_text,
                                             kernel_size=(kernel_sizes_text[i], dim_text),
                                             stride=1,
                                             padding=0,
                                             dilation=1,
                                             groups=1,
                                             bias=False) for i in range(self.num_kernels_text))
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc_w_ts = [torch.zeros(self.num_class * self.k_ts, self.num_class, dtype=torch.float32) for _ in range(self.num_kernels_ts)]
        for i in range(self.num_kernels_ts):
            for j in range(self.num_class):
                self.fc_w_ts[i][j*k_ts:(j+1)*k_ts, j] = 1.0
        self.fc_w_text = [torch.zeros(self.num_class * self.k_text, self.num_class, dtype=torch.float32) for _ in range(self.num_kernels_text)]
        for i in range(self.num_kernels_text):
            for j in range(self.num_class):
                self.fc_w_text[i][j*k_text:(j+1)*k_text, j] = 1.0
        self.fc_w = torch.cat(self.fc_w_ts + self.fc_w_text, dim=0).to(device)  # (num_kernels_ts * num_class * k_ts + num_kernels_text * num_class * k_text, num_class)
        self.fc = nn.Linear(self.num_kernels_ts * self.num_class * self.k_ts + self.num_kernels_text * self.num_class * self.k_text, self.num_class, bias=False)
        self.fc.weight.data = torch.transpose(self.fc_w, 0, 1)
        self.prottps_ts = nn.ParameterList([nn.Parameter(glorot([self.num_class * self.k_ts, cout_ts]), requires_grad=True) for _ in range(self.num_kernels_ts)])
        self.prottps_text = nn.ParameterList([nn.Parameter(glorot([self.num_class * self.k_text, cout_text]), requires_grad=True) for _ in range(self.num_kernels_text)])

    def init_prottp_ts(self, x, y, msk, device):
        '''
        :param x: (batch, len, dim)
        :param y: (batch,)
        :param msk: num_kernels * [(batch, len - kernel_size + 1)]
        :param device:
        :return:
        '''
        x_convs = self.convolution_ts(x)  # num_kernels * [(batch, len - kernel_size + 1, cout)]
        for i in range(self.num_kernels_ts):
            for j in range(self.num_class):
                x = x_convs[i][y == j, :, :]  # (batch_j, (len - kernel_size + 1), cout)
                x = x.reshape(-1, x.size(2))  # (batch_j * (len - kernel_size + 1), cout)
                msk_ij = msk[i][y == j, :]  # (batch_j, len - kernel_size + 1)
                msk_ij = msk_ij.reshape(-1)  # (batch_j * (len - kernel_size + 1),)
                x = x[msk_ij == 0, :]
                idx = torch.randperm(x.size(0))[:self.k_ts]
                x = x[idx, :]  # (k, cout)
                if x.size(0) != 0.0:
                    self.prottps_ts[i].data[j*self.k_ts:(j+1)*self.k_ts, :] = x + x * torch.rand(x.size(0), x.size(1)).to(device) * 1e-3
            
        return None
    
    def init_prottp_text(self, x, y, msk, device):
        '''
        :param x: (batch, len, dim)
        :param y: (batch,)
        :param msk: num_kernels * [(batch, len - kernel_size + 1)]
        :param device:
        :return:
        '''
        x_convs = self.convolution_text(x)  # num_kernels * [(batch, len - kernel_size + 1, cout)]
        for i in range(self.num_kernels_text):
            for j in range(self.num_class):
                x = x_convs[i][y == j, :, :]  # (batch_j, (len - kernel_size + 1), cout)
                x = x.reshape(-1, x.size(2))  # (batch_j * (len - kernel_size + 1), cout)
                msk_ij = msk[i][y == j, :]  # (batch_j, len - kernel_size + 1)
                msk_ij = msk_ij.reshape(-1)  # (batch_j * (len - kernel_size + 1),)
                x = x[msk_ij == 0, :]
                idx = torch.randperm(x.size(0))[:self.k_text]
                x = x[idx, :]  # (k, cout)
                if x.size(0) != 0.0:
                    self.prottps_text[i].data[j*self.k_text:(j+1)*self.k_text, :] = x + x * torch.rand(x.size(0), x.size(1)).to(device) * 1e-3
            
        return None

    def convolution_ts(self, x):
        '''
        :param x: (batch, len, dim)
        :return:
        '''
        x = x.unsqueeze(1)  # (batch, cin=1, len, dim)
        x_new = []
        for conv in self.convs_ts:
            x_conv = torch.tanh(conv(x)).squeeze(3)  # (batch, cout, len - kernel_size + 1)
            # x_conv = F.relu(conv(x)).squeeze(3)  # (batch, cout, len - kernel_size + 1)
            # x_conv = F.softplus(conv(x), beta=1, threshold=20)  # (batch, cout, len - kernel_size + 1)
            x_new.append(torch.transpose(x_conv, 1, 2))  # (batch, len - kernel_size + 1, cout)
        return x_new
    
    def convolution_text(self, x):
        '''
        :param x: (batch, len, dim)
        :return:
        '''
        x = self.text_emb_fc(x)
        x = self.text_dropout(x)
        x = x.unsqueeze(1)  # (batch, cin=1, len, dim)
        x_new = []
        for conv in self.convs_text:
            x_conv = torch.tanh(conv(x)).squeeze(3)  # (batch, cout, len - kernel_size + 1)
            # x_conv = F.relu(conv(x)).squeeze(3)  # (batch, cout, len - kernel_size + 1)
            # x_conv = F.softplus(conv(x), beta=1, threshold=20)  # (batch, cout, len - kernel_size + 1)
            x_new.append(torch.transpose(x_conv, 1, 2))  # (batch, len - kernel_size + 1, cout)
        return x_new
    
 

    def loss_ts(self, logit, y_one_hot, dists, lmd0, lmd1, lmd2, lmd3, lmd4, dmin):
        '''
        :param logit: (batch, num_class)
        :param y_one_hot: (batch, num_class)
        :param dists: num_kernels * [(batch, (len - kernel_size + 1), num_class, k)]
        :param lmd0:
        :param lmd1:
        :param lmd2:
        :param lmd3:
        :param dmin:
        :return:
        '''
        batchsz = y_one_hot.size(0)
        ls0 = F.binary_cross_entropy(logit, y_one_hot, reduction='mean')

        if lmd1 != 0.0:
            ls1 = 0.0
            for dist in dists:
                mindist = dist.min(3).values.min(1).values  # (batch, num_class)
                ls1 += (mindist * y_one_hot).sum()
            ls1 = ls1 / (self.num_kernels_ts * batchsz)
        else:
            ls1 = ls0

        if lmd2 != 0.0:
            ls2 = 0.0
            for dist in dists:
                mindist = dist.min(1).values  # (batch, num_class, k)
                for i in range(self.num_class):
                    if (y_one_hot[:, i] == 1).sum() > 0.0:
                        ls2 += mindist[y_one_hot[:, i] == 1, i, :].min(0).values.sum()
            ls2 = ls2 / (self.num_kernels_ts * self.num_class * self.k_ts)
        else:
            ls2 = ls0

        if lmd3 != 0.0:
            ls3 = 0.0
            for p in self.prottps_ts:
                pdist = (((p.unsqueeze(0) - p.unsqueeze(1)) ** 2).sum(2) + 1e-8) ** 0.5  # (num_class * k, num_class * k)
                ls3 += (((dmin - pdist).clamp(0).triu(1)) ** 2).sum()
            ls3 = ls3 / (self.num_kernels_ts * self.num_class * self.k_ts * (self.num_class * self.k_ts - 1) * 0.5)
        else:
            ls3 = ls0

        if lmd4 != 0.0:
            ls4 = 0.0
            for p in self.prottps_ts:
                pdist = (((p.unsqueeze(0) - p.unsqueeze(1)) ** 2).sum(2) + 1e-8) ** 0.5  # (num_class * k, num_class * k)
                pdist = (dmin - pdist).clamp(0)  # (num_class * k, num_class * k)
                for i in range(self.num_class):
                    ls4 += pdist[i*self.num_class:(i+1)*self.num_class, (i+1)*self.num_class:].sum()
            ls4 = ls4 / (self.num_kernels_ts + self.num_class * (self.num_class - 1) * 0.5 * self.k_ts * self.k_ts)
        else:
            ls4 = ls0

        ls = lmd0 * ls0 + lmd1 * ls1 + lmd2 * ls2 + lmd3 * ls3 + lmd4 * ls4
        return ls, ls0, ls1, ls2, ls3, ls4
    
    def loss_text(self, logit, y_one_hot, dists, lmd0, lmd1, lmd2, lmd3, lmd4, dmin):
        '''
        :param logit: (batch, num_class)
        :param y_one_hot: (batch, num_class)
        :param dists: num_kernels * [(batch, (len - kernel_size + 1), num_class, k)]
        :param lmd0:
        :param lmd1:
        :param lmd2:
        :param lmd3:
        :param dmin:
        :return:
        '''
        batchsz = y_one_hot.size(0)
        # ls0 = F.cross_entropy(logit, y_one_hot.argmax(1), reduction='mean')
        ls0 = F.binary_cross_entropy(logit, y_one_hot, reduction='mean')

        if lmd1 != 0.0:
            ls1 = 0.0
            for dist in dists:
                mindist = dist.min(3).values.min(1).values  # (batch, num_class)
                ls1 += (mindist * y_one_hot).sum()
            ls1 = ls1 / (self.num_kernels_text * batchsz)
        else:
            ls1 = ls0

        if lmd2 != 0.0:
            ls2 = 0.0
            for dist in dists:
                mindist = dist.min(1).values  # (batch, num_class, k)
                for i in range(self.num_class):
                    if (y_one_hot[:, i] == 1).sum() > 0.0:
                        ls2 += mindist[y_one_hot[:, i] == 1, i, :].min(0).values.sum()
            ls2 = ls2 / (self.num_kernels_text * self.num_class * self.k_text)
        else:
            ls2 = ls0

        if lmd3 != 0.0:
            ls3 = 0.0
            for p in self.prottps_text:
                pdist = (((p.unsqueeze(0) - p.unsqueeze(1)) ** 2).sum(2) + 1e-8) ** 0.5  # (num_class * k, num_class * k)
                ls3 += (((dmin - pdist).clamp(0).triu(1)) ** 2).sum()
            ls3 = ls3 / (self.num_kernels_text * self.num_class * self.k_text * (self.num_class * self.k_text - 1) * 0.5)
        else:
            ls3 = ls0

        if lmd4 != 0.0:
            ls4 = 0.0
            for p in self.prottps_text:
                pdist = (((p.unsqueeze(0) - p.unsqueeze(1)) ** 2).sum(2) + 1e-8) ** 0.5  # (num_class * k, num_class * k)
                pdist = (dmin - pdist).clamp(0)  # (num_class * k, num_class * k)
                for i in range(self.num_class):
                    ls4 += pdist[i*self.num_class:(i+1)*self.num_class, (i+1)*self.num_class:].sum()
            ls4 = ls4 / (self.num_kernels_text + self.num_class * (self.num_class - 1) * 0.5 * self.k_text * self.k_text)
        else:
            ls4 = ls0

        ls = lmd0 * ls0 + lmd1 * ls1 + lmd2 * ls2 + lmd3 * ls3 + lmd4 * ls4
        return ls, ls0, ls1, ls2, ls3, ls4

    def projection_ts(self, x, y, msk):
        '''
        :param x: (batch, len, dim)
        :param y: (batch,)
        :param msk: num_kernels * [(batch, len - kernel_size + 1)]
        :return:
        '''
        n = x.size(0)
        x_convs = self.convolution_ts(x)  # num_kernels * [(batch, len - kernel_size + 1, cout)]
        p_seq = []
        p_subseq_idx = []
        for i in range(self.num_kernels_ts):
            x_conv_i = x_convs[i]  # (batch, (len - kernel_size + 1), cout)
            x_conv_i_len = x_conv_i.size(1)
            x_conv_i = x_conv_i.reshape(-1, x_conv_i.size(2))  # (batch * (len - kernel_size + 1), cout)
            p = self.prottps_ts[i]  # (num_class * k, cout)
            dist = ((x_conv_i ** 2).sum(1).reshape(x_conv_i.size(0), 1) - 2 * torch.mm(x_conv_i, torch.transpose(p, 0, 1))) + (p ** 2).sum(1).reshape(1, p.size(0))  # (batch * (len - kernel_size + 1), num_class * k)
            dist = dist + msk[i].reshape(-1, 1)  # (batch * (len - kernel_size + 1), num_class * k)
            dist = dist.reshape(n, -1, self.num_class * self.k_ts)  # (batch, (len - kernel_size + 1), num_class * k)
            p_new = []
            p_seq_i = []
            p_subseq_idx_i = []

            for j in range(self.num_class):
                idx_j = (y == j).nonzero().reshape(-1)  # (batch_j,)
                dist_j = dist[idx_j, :, j*self.k_ts:(j+1)*self.k_ts]  # (batch_j, (len - kernel_size + 1), k)
                dist_j = dist_j.reshape(-1, self.k_ts)  # (batch_j * (len - kernel_size + 1), k)
                idx1 = torch.argmin(dist_j, dim=0)  # (k,)
                idx2 = idx1 // x_conv_i_len  # (k,) batch index
                idx3 = idx1 - idx2 * x_conv_i_len  # (k,) seq index
                p_new.append(x_conv_i[(idx_j[idx2] * x_conv_i_len + idx3), :])  # num_class * [(k, cout)]
                idx4 = idx_j[idx2]  # (k,)
                idx5 = torch.stack([idx3 + l for l in range(self.convs_ts[i].kernel_size[0])])  # (kernel_size, k)
                idx5 = torch.transpose(idx5, 0, 1)  # (k, kernel_size)
                p_seq_i.append(x[idx4, :, :])  # num_class * [(k, len, dim)]
                p_subseq_idx_i.append(idx5)  # num_class * [(k, kernel_size)]

            self.prottps_ts[i].data = torch.cat(p_new, dim=0)  # num_kernels * [(num_class * k, cout),]
            p_seq.append(torch.cat(p_seq_i, dim=0))  # num_kernels * [(num_class * k, len, dim),]
            p_subseq_idx.append(torch.cat(p_subseq_idx_i, dim=0))  # num_kernels * [(num_class * k, kernel_size),]
        return p_seq, p_subseq_idx
    
    def projection_text(self, x, y, msk, x_token):
        '''
        :param x: (batch, len, dim)
        :param y: (batch,)
        :param msk: num_kernels * [(batch, len - kernel_size + 1)]
        :return:
        '''
        n = x.size(0)
        x_convs = self.convolution_text(x)  # num_kernels * [(batch, len - kernel_size + 1, cout)]
        p_seq = []
        p_seq_token = []
        p_subseq_idx = []
        
        for i in range(self.num_kernels_text):
            x_conv_i = x_convs[i]  # (batch, (len - kernel_size + 1), cout)
            x_conv_i_len = x_conv_i.size(1) # (len - kernel_size + 1): number of segments
            x_conv_i = x_conv_i.reshape(-1, x_conv_i.size(2))  # (batch * (len - kernel_size + 1), cout)
            p = self.prottps_text[i]  # (num_class * k, cout)
            dist = ((x_conv_i ** 2).sum(1).reshape(x_conv_i.size(0), 1) - 2 * torch.mm(x_conv_i, torch.transpose(p, 0, 1))) + (p ** 2).sum(1).reshape(1, p.size(0))  # (batch * (len - kernel_size + 1), num_class * k)
            dist = dist + msk[i].reshape(-1, 1)  # (batch * (len - kernel_size + 1), num_class * k)
            dist = dist.reshape(n, -1, self.num_class * self.k_text)  # (batch, (len - kernel_size + 1), num_class * k)
            p_new = []
            p_seq_i = []
            p_seq_token_i = []
            p_subseq_idx_i = []

            for j in range(self.num_class):
                idx_j = (y == j).nonzero().reshape(-1)  # (batch_j,) select instance with label j from batch
                dist_j = dist[idx_j, :, j*self.k_text:(j+1)*self.k_text]  # (batch_j, (len - kernel_size + 1), k)
                dist_j = dist_j.reshape(-1, self.k_text)  # (batch_j * (len - kernel_size + 1), k)
                idx1 = torch.argmin(dist_j, dim=0)  # (k,)
                idx2 = idx1 // x_conv_i_len  # (k,) instance index in batch_j
                idx3 = idx1 - idx2 * x_conv_i_len  # (k,) seq index
                p_new.append(x_conv_i[(idx_j[idx2] * x_conv_i_len + idx3), :])  # num_class * [(k, cout)]
                idx4 = idx_j[idx2]  # (k,)
                idx5 = torch.stack([idx3 + l for l in range(self.convs_text[i].kernel_size[0])])  # (kernel_size, k)
                idx5 = torch.transpose(idx5, 0, 1)  # (k, kernel_size)
                p_seq_i.append(x[idx4, :, :])  # num_class * [(k, len, dim)]
                p_seq_token_i.append(x_token[idx4, :])
                p_subseq_idx_i.append(idx5)  # num_class * [(k, kernel_size)]

            self.prottps_text[i].data = torch.cat(p_new, dim=0)  # num_kernels * [(num_class * k, cout),]
            p_seq.append(torch.cat(p_seq_i, dim=0))  # num_kernels * [(num_class * k, len, dim),]
            p_subseq_idx.append(torch.cat(p_subseq_idx_i, dim=0))  # num_kernels * [(num_class * k, kernel_size),]
            p_seq_token.append(torch.cat(p_seq_token_i, dim=0))
            
        return p_seq, p_subseq_idx, p_seq_token

    def forward(self, x_ts, x_text, msk_ts, msk_text):
        '''
        :param x: (batch, len, dim)
        :param msk: num_kernels * [(batch, len - kernel_size + 1)]
        :return:
        '''
        n = x_ts.size(0)
        x_convs_ts = self.convolution_ts(x_ts)  # num_kernels * [(batch, len - kernel_size_ts + 1, cout_ts)]
        x_convs_text = self.convolution_text(x_text)  # num_kernels * [(batch, len - kernel_size_ts + 1, cout_ts)]
        logits = []
        dists_ts = []
        dists_text = []
        for i in range(self.num_kernels_ts):
            x = x_convs_ts[i]  # (batch, (len - kernel_size_ts + 1), cout_ts)
            x = x.reshape(-1, x.size(2))  # (batch * (len - kernel_size_ts + 1), cout)
            p = self.prottps_ts[i]  # (num_class * k_ts, cout_ts)
            logit = ((x ** 2).sum(1).reshape(x.size(0), 1) - 2 * torch.mm(x, torch.transpose(p, 0, 1))) + (p ** 2).sum(1).reshape(1, p.size(0))  # (batch * (len - kernel_size_ts + 1), num_class * k_ts)
            logit = logit + msk_ts[i].reshape(-1, 1)  # (batch * (len - kernel_size_ts + 1), num_class * k_ts)
            dists_ts.append(logit.reshape(n, -1, self.num_class, self.k_ts))  # num_kernels * [(batch, (len - kernel_size_ts + 1), num_class, k_ts)]
            logit = torch.exp(-logit)  # (batch * (len - kernel_size_ts + 1), num_class * k_ts)
            logit = logit.reshape(n, -1, self.num_class * self.k_ts)  # (batch, (len - kernel_size_text + 1), num_class * k_ts)
            logit = torch.transpose(logit, 1, 2)  # (batch, num_class * k_ts, len - kernel_size_ts + 1)
            logit = F.max_pool1d(logit, logit.size(2)).squeeze(2)  # (batch, num_class * k_ts)
            logits.append(logit)  # num_kernels_ts * [(batch, num_class * k_ts)]
            
        for i in range(self.num_kernels_text):
            x = x_convs_text[i]  # (batch, (len - kernel_size_text + 1), cout_text)
            x = x.reshape(-1, x.size(2))  # (batch * (len - kernel_size_text + 1), cout_text)
            p = self.prottps_text[i]  # (num_class * k_ts, cout_ts)
            logit = ((x ** 2).sum(1).reshape(x.size(0), 1) - 2 * torch.mm(x, torch.transpose(p, 0, 1))) + (p ** 2).sum(1).reshape(1, p.size(0))  # (batch * (len - kernel_size_text + 1), num_class * k_text)
            logit = logit + msk_text[i].reshape(-1, 1)  # (batch * (len - kernel_size_ts + 1), num_class * k_ts)
            dists_text.append(logit.reshape(n, -1, self.num_class, self.k_text))  # num_kernels * [(batch, (len - kernel_size_text + 1), num_class, k_text)]
            logit = torch.exp(-logit)  # (batch * (len - kernel_size_text + 1), num_class * k_text)
            logit = logit.reshape(n, -1, self.num_class * self.k_text)  # (batch, (len - kernel_size_text + 1), num_class * k_text)
            logit = torch.transpose(logit, 1, 2)  # (batch, num_class * k_ts, len - kernel_size_text + 1)
            logit = F.max_pool1d(logit, logit.size(2)).squeeze(2)  # (batch, num_class * k_ts)
            logits.append(logit)  # num_kernels_ts * [(batch, num_class * k_ts)]

        logit = torch.cat(logits, dim=1)  # (batch, num_kernels_ts * num_class * k_ts + num_kernels_text * num_class * k_text)
        logit = self.dropout(logit)  # (batch, num_kernels_ts * num_class * k_ts + num_kernels_text * num_class * k_text)
        logit = self.fc(logit)  # (batch, num_class)
        logit = torch.softmax(logit, dim=1)
        return logit, dists_ts, dists_text