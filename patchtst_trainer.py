import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
from sklearn.metrics import f1_score, roc_auc_score

class Trainer:
    def __init__(self, model, config, train_loader, val_loader, test_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.num_epochs)
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, 'patchtst_checkpoint.pth')
        
        self.best_val_f1 = -1
        self.patience = self.config.patience
        self.early_stopping_counter = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.long().to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x, None, None, None)
            loss = self.loss_fn(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        all_labels, all_preds, all_probs, all_logits = [], [], [], []
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                outputs = self.model(batch_x, None, None, None)
                loss = self.loss_fn(outputs, batch_y)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                probs = torch.softmax(outputs, dim=1)

                all_labels.extend(batch_y.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_logits.append(outputs.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
        f1 = f1_score(all_labels, all_preds)
        return total_loss / len(data_loader), accuracy, f1, all_labels, all_preds, all_probs, all_logits

    def train(self):
        print("Starting training...")
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc, val_f1, _, _, _, _ = self.evaluate(self.val_loader)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f'Epoch {epoch+1}/{self.config.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f} | LR: {current_lr:.6f}')

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.early_stopping_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"New best model saved to {self.checkpoint_path} with F1 score: {val_f1:.4f}")
            else:
                self.early_stopping_counter += 1
                print(f"Validation F1 score did not improve. Early stopping counter: {self.early_stopping_counter}/{self.patience}")
                if self.early_stopping_counter >= self.patience:
                    print("Early stopping triggered.")
                    break
        print("Training complete.")

        # Save logits and probabilities for train and validation sets
        print("Saving logits and probabilities...")
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        
        # Correctly unpack all 7 return values
        _, _, _, _, _, train_probs, train_logits = self.evaluate(self.train_loader)
        _, _, _, _, _, val_probs, val_logits = self.evaluate(self.val_loader)

        np.save(os.path.join(self.checkpoint_dir, 'train_logits.npy'), train_logits)
        np.save(os.path.join(self.checkpoint_dir, 'train_probs.npy'), train_probs)
        np.save(os.path.join(self.checkpoint_dir, 'val_logits.npy'), val_logits)
        np.save(os.path.join(self.checkpoint_dir, 'val_probs.npy'), val_probs)

        print(f"Logits and probabilities saved in {self.checkpoint_dir}")

    def test(self):
        print("Starting testing...")
        try:
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            print(f"Model loaded from {self.checkpoint_path}")
        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {self.checkpoint_path}. Please train the model first.")
            return

        # Correctly unpack all 7 return values
        test_loss, test_acc, test_f1, test_labels, test_preds, test_probs, _ = self.evaluate(self.test_loader)
        auc = roc_auc_score(test_labels, test_probs[:, 1])

        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | F1: {test_f1:.4f} | AUC: {auc:.4f}')
        print("Testing complete.")
