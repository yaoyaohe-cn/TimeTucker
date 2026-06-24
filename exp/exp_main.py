from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import TimeTucker
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.use_orthogonal = args.use_orthogonal
        
    def _build_model(self):
        model_dict = {'TimeTucker': TimeTucker}
        model = model_dict[self.args.model].Model(self.args).float()
        print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _is_single_input_model(self):
        return self.args.model in {'TimeTucker'} or any(
            substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}
        )

    def _unpack_single_input_output(self, outputs):
        if isinstance(outputs, tuple):
            return outputs[0], outputs[1]
        return outputs, 0

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        if self.args.loss == "mae": return nn.L1Loss()
        elif self.args.loss == "smooth": return nn.SmoothL1Loss()
        else: return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x = batch[0].float().to(self.device)
                batch_y = batch[1].float()
                if len(batch) == 4:
                    batch_x_mark = batch[2].float().to(self.device)
                    batch_y_mark = batch[3].float().to(self.device)
                else:
                    batch_x_mark, batch_y_mark = None, None

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self._is_single_input_model():
                            outputs, _ = self._unpack_single_input_output(self.model(batch_x))
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            if self.args.output_attention: outputs = outputs[0]
                else:
                    if self._is_single_input_model():
                        outputs, _ = self._unpack_single_input_output(self.model(batch_x))
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.output_attention: outputs = outputs[0]
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs.detach().cpu(), batch_y.detach().cpu())
                total_loss.append(loss)
                
        self.model.train()
        return np.average(total_loss)

    def vali_from_setting(self, setting):
        best_model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        vali_data, vali_loader = self._get_data(flag='val')
        return float(self.vali(vali_data, vali_loader, self._select_criterion()))

    def train(self, setting, optunaTrialReport=None):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = None
        if self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)
        max_memory = -1
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            ortho_loss_record = []
            self.model.train()
            epoch_time = time.time()
            
            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch[0].float().to(self.device)
                batch_y = batch[1].float().to(self.device)
                if len(batch) == 4:
                    batch_x_mark = batch[2].float().to(self.device)
                    batch_y_mark = batch[3].float().to(self.device)
                else:
                    batch_x_mark, batch_y_mark = None, None

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                orthogonal_loss = 0
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self._is_single_input_model():
                            outputs, orthogonal_loss = self._unpack_single_input_output(self.model(batch_x))
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            if self.args.output_attention: outputs = outputs[0]

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                else:
                    if self._is_single_input_model():
                        outputs, orthogonal_loss = self._unpack_single_input_output(self.model(batch_x))
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.output_attention: outputs = outputs[0]
                            
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())
                if isinstance(orthogonal_loss, torch.Tensor):
                    ortho_loss_record.append(orthogonal_loss.item())
                else:
                    ortho_loss_record.append(orthogonal_loss)

                if self.args.use_amp:
                    back_loss = loss + self.args.orthogonal_weight * orthogonal_loss
                    scaler.scale(back_loss).backward()
                    if self.args.clip > 0:
                        scaler.unscale_(model_optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    back_loss = loss + self.args.orthogonal_weight * orthogonal_loss 
                    back_loss.backward()
                    if self.args.clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                    model_optim.step()
                    
                if self.device.type == 'cuda':
                    max_memory = max(max_memory, torch.cuda.max_memory_allocated(device=self.device) / 1024 ** 2)
                
                if self.args.lradj == 'TST' and scheduler is not None:
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

                if (i + 1) % 100 == 0:
                    curr_ortho = ortho_loss_record[-1] if len(ortho_loss_record) > 0 else 0
                    print("\titers: {}, epoch: {} | loss: {:.7f} | ortho_loss: {:.7f}".format(i + 1, epoch + 1, loss.item(), curr_ortho))
                    
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {} | Train Loss: {:.7f} Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss))
                
            if optunaTrialReport is not None:
                optunaTrialReport.report(float(vali_loss), epoch)
                if optunaTrialReport.should_prune():
                    import optuna
                    raise optuna.exceptions.TrialPruned()

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(path + '/checkpoint.pth')) 
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))

        preds, trues = [], []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x = batch[0].float().to(self.device)
                batch_y = batch[1].float().to(self.device)
                if len(batch) == 4:
                    batch_x_mark = batch[2].float().to(self.device)
                    batch_y_mark = batch[3].float().to(self.device)
                else:
                    batch_x_mark, batch_y_mark = None, None

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self._is_single_input_model():
                            outputs, _ = self._unpack_single_input_output(self.model(batch_x))
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            if self.args.output_attention: outputs = outputs[0]
                else:
                    if self._is_single_input_model():
                        outputs, _ = self._unpack_single_input_output(self.model(batch_x))
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.output_attention: outputs = outputs[0]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))

        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        
        return mse, mae

    def predict(self, setting, load=False):
        return
