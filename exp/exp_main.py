from exp.exp_basic import Exp_Basic
from models.TimeScaler import TimeScaler
from utils.losses import StructuredLoss
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from data_provider.data_factory import data_provider

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import optuna
import warnings

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.min_test_loss = np.inf
        
    def _build_model(self):
        # TimeScaler Initialization
        model = TimeScaler(
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            c_in=self.args.c_in,
            d_model=self.args.d_model,
            dropout=self.args.dropout,
            wave_level=self.args.level,
            wave_basis=self.args.wavelet,
            device=self.device
        ).float()
            
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        
        model_optim = optim.Adam(self.model.parameters(), 
                                 lr=self.args.learning_rate, 
                                 weight_decay=self.args.weight_decay)
        return model_optim
    
    def _select_criterion(self):
            loss_name = getattr(self.args, 'loss', 'smoothL1')
            
            criterion = StructuredLoss(
                loss_name=loss_name,
                alpha=1.0, 
                beta=1.0, 
                gamma=1.0
            ) 
            return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        preds_mean, trues = [], []

        with torch.no_grad():
            for batch_x, batch_y in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # Validation: Just get the reconstruction
                pred = self.model(batch_x, return_decomposition=False)

                # Use MSE for validation monitoring (Early Stopping)
                loss = torch.nn.functional.mse_loss(pred, batch_y)
                total_loss.append(loss.item())

                preds_mean.append(pred.detach().cpu())
                trues.append(batch_y.detach().cpu())

            preds_mean = torch.cat(preds_mean).numpy()
            trues = torch.cat(trues).numpy()

            mae, mse, rmse, mape, mspe = metric(preds_mean, trues)
            self.model.train()

            # Return MSE as the loss metric for EarlyStopping
            return mse, mae

    def vali_from_setting(self, setting):
        """
        Load the best model from checkpoint and evaluate on validation set.
        Used by Tuner for hyperparameter optimization.
        """
        # Load best model checkpoint
        print(f'Loading best model from checkpoint: {setting}')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        # Get validation data and evaluate
        vali_data, vali_loader = self._get_data(flag='val')
        criterion = self._select_criterion()
        
        val_mse, val_mae = self.vali(vali_data, vali_loader, criterion)
        print(f'Validation MSE: {val_mse:.6f}, MAE: {val_mae:.6f}')
        
        return val_mse, val_mae

    def train(self, setting, optunaTrialReport=None):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion() # StructuredLoss

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler(init_scale=1024)
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # Training with Structured Loss
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        pred, pred_yl, pred_yh = self.model(batch_x, return_decomposition=True)
                        
                        raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                        
                        loss = criterion(
                            pred=pred, 
                            true=batch_y, 
                            pred_yl=pred_yl, 
                            pred_yh_list=pred_yh, 
                            odb_module=raw_model.odb, 
                            revin_module=raw_model.global_revin
                        )
                        
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    pred, pred_yl, pred_yh = self.model(batch_x, return_decomposition=True)
                    
                    raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                    
                    loss = criterion(
                        pred=pred, 
                        true=batch_y, 
                        pred_yl=pred_yl, 
                        pred_yh_list=pred_yh, 
                        odb_module=raw_model.odb, 
                        revin_module=raw_model.global_revin
                    )
                    
                    loss.backward()
                    model_optim.step()

                train_loss.append(loss.item())

            print("Epoch {}: cost time: {:.2f} sec".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_mse, vali_mae = self.vali(vali_data, vali_loader, criterion)
            test_mse, test_mae = self.vali(test_data, test_loader, criterion)

            # [Restored] Optuna Reporting & Pruning
            if optunaTrialReport is not None:
                optunaTrialReport.report(test_mse, epoch)
                if optunaTrialReport.should_prune():
                    raise optuna.exceptions.TrialPruned()

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.5f} Vali MSE: {3:.5f} Test MSE: {4:.5f}".format(
                epoch + 1, train_steps, train_loss, vali_mse, test_mse))
            
            # Early Stopping monitors Validation MSE
            early_stopping(vali_mse, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optim, None, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()
        preds = []
        trues = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred = self.model(batch_x, return_decomposition=False)
                
                preds.append(pred.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
       
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return mse, mae
