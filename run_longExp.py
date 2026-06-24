import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes', 'y'):
        return True
    if value.lower() in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='SparseTSF & TimeTucker for Time Series Forecasting')

# basic config
parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task name')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='TimeTucker', help='model name')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# forecasting task
parser.add_argument('--seq_len', type=int, default=720, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# TimeTucker / SparseTSF
parser.add_argument('--period_len', type=int, default=24, help='period length')
parser.add_argument('--r_n', type=int, default=6, help='TimeTucker segment rank')
parser.add_argument('--r_c', type=int, default=16, help='TimeTucker channel rank')
parser.add_argument('--r_p', type=int, default=8, help='TimeTucker period rank')
parser.add_argument('--use_period_norm', type=int, default=1, help='norm')
parser.add_argument('--use_revin', type=int, default=1, help='RevIN normalization')
parser.add_argument('--share_factors', type=int, default=1, help='share encoder/decoder factors')
parser.add_argument('--use_orthogonal', type=int, default=1, help='orthogonal')
parser.add_argument('--orthogonal_weight', type=float, default=0.005, help='orthogonal weight')

# PatchTST / Formers (Retained for compatibility)
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='padding on the end')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition')
parser.add_argument('--kernel_size', type=int, default=7, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head')
parser.add_argument('--embed_type', type=int, default=0, help='embedding type')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') 
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false', default=True, help='use distilling in encoder')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='learned', help='time features encoding')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', default=False, help='output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--clip', type=float, default=1.0, help='gradient clipping norm, 0 means no clipping')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# hyperparameter optimization
parser.add_argument('--use_hyperParam_optim', action='store_true', default=False, help='use Optuna tuner')
parser.add_argument('--optuna_trial_num', type=int, default=30, help='number of Optuna trials')
parser.add_argument('--optuna_n_startup_trials', type=int, default=8, help='startup trials before Optuna pruning')
parser.add_argument('--optuna_seed', type=int, default=2026, help='random seed for Optuna tuning')
parser.add_argument('--n_jobs', type=int, default=1, help='parallel Optuna jobs (MUST BE 1 for PyTorch global seed safety)')
parser.add_argument('--optuna_results_dir', type=str, default='./Output', help='Optuna output directory')

# GPU
parser.add_argument('--use_gpu', type=str2bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', type=int, help='use multiple gpus', default=0)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

fix_seed_list = range(args.optuna_seed, args.optuna_seed + 10)
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training and args.use_hyperParam_optim:
    from utils.Tuner import Tuner
    tuner = Tuner(ranSeed=args.optuna_seed, n_jobs=args.n_jobs)
    tuner.tune(args)
elif args.is_training:
    for ii in range(args.itr):
        random.seed(fix_seed_list[ii])
        torch.manual_seed(fix_seed_list[ii])
        np.random.seed(fix_seed_list[ii])
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_{}_{}_seed{}'.format(
            args.model_id, args.model, args.data, args.features,
            args.seq_len, args.pred_len, args.des, ii, fix_seed_list[ii])

        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_{}_{}_seed{}'.format(
        args.model_id, args.model, args.data, args.features,
        args.seq_len, args.pred_len, args.des, ii, fix_seed_list[ii])

    exp = Exp(args)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()