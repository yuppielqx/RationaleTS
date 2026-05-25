import numpy as np
import pickle as pkl
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from PatchTST import Model as PatchTST
from patchtst_trainer import Trainer
import argparse

def load_and_process_data(args):
    """Loads data, preprocesses it, and returns DataLoaders."""
    data_path = './dataset/weather_ny/'
    with open(os.path.join(data_path, 'indices.pkl'), 'rb') as f:
        indices = pkl.load(f)
    with open(os.path.join(data_path, 'rain.pkl'), 'rb') as f:
        rain = pkl.load(f)
    with open(os.path.join(data_path, 'time_series_ny.pkl'), 'rb') as pklfile:
        ts = pkl.load(pklfile)

    data_size = len(indices)
    num_train = int(data_size * 0.6)
    num_test = int(data_size * 0.2)
    num_vali = data_size - num_train - num_test

    seq_len_day = 1

    train_idx = np.arange(num_train - seq_len_day)
    val_idx = np.arange(num_train - seq_len_day, num_train + num_vali - seq_len_day)
    test_idx = np.arange(num_train + num_vali - seq_len_day, num_train + num_vali + num_test - seq_len_day)

    train_data, train_label = [], []
    val_data, val_label = [], []
    test_data, test_label = [], []

    scaler = StandardScaler()
    scaler.fit(ts[:indices[train_idx[-1]]+args.seq_len,:])
    ts1 = scaler.transform(ts)

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    for i in train_idx:
        idx = indices[i]
        train_data.append(ts1[idx:idx + args.seq_len, :])
        train_label.append(int(rain[i + args.pred_len]))

    for i in val_idx:
        idx = indices[i]
        val_data.append(ts1[idx:idx + args.seq_len, :])
        val_label.append(int(rain[i + args.pred_len]))

    for i in test_idx:
        idx = indices[i]
        test_data.append(ts1[idx:idx + args.seq_len, :])
        test_label.append(int(rain[i + args.pred_len]))

    x_tr_ts, y_tr = np.array(train_data), np.array(train_label)
    x_va_ts, y_va = np.array(val_data), np.array(val_label)
    x_te_ts, y_te = np.array(test_data), np.array(test_label)

    train_dataset = TensorDataset(torch.from_numpy(x_tr_ts), torch.from_numpy(y_tr))
    val_dataset = TensorDataset(torch.from_numpy(x_va_ts), torch.from_numpy(y_va))
    test_dataset = TensorDataset(torch.from_numpy(x_te_ts), torch.from_numpy(y_te))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def main(args):
    train_loader, val_loader, test_loader = load_and_process_data(args)

    patchtst_model = PatchTST(args)
    trainer = Trainer(patchtst_model, args, train_loader, val_loader, test_loader)

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        trainer.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test the PatchTST model.')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode to run: train or test')
    
    # Model Architecture
    parser.add_argument('--task_name', type=str, default='classification', help='Task name')
    parser.add_argument('--seq_len', type=int, default=24, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=1, help='Prediction sequence length')
    parser.add_argument('--d_model', type=int, default=64, help='Dimension of model')
    parser.add_argument('--n_heads', type=int, default=16, help='Number of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--enc_in', type=int, default=5, help='Encoder input size')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes')
    parser.add_argument('--factor', type=int, default=1, help='Attention factor')

    # Training Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--checkpoint_dir', type=str, default='expl_results/weather_ny', help='Path to save checkpoints')
    parser.add_argument('--seed', type=int, default=10, help='Random seed')

    args = parser.parse_args()
    main(args)
