"""
CLI entry for Bayesian hyper-parameter search on TimeTucker.

Mirrors run_longExp.py's arg surface so existing dataset scripts can reuse
the same flags, then delegates search/log to utils.bayes_tuner.
"""

import argparse

import torch

from utils.bayes_tuner import TimeTuckerBayesTuner


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes', 'y'):
        return True
    if value.lower() in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Bayesian HP search for TimeTucker')

# ---- forecasting task ----
parser.add_argument('--task_name', type=str, default='long_term_forecast')
parser.add_argument('--model', type=str, default='TimeTucker')
parser.add_argument('--model_id', type=str, default='bayes')
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--root_path', type=str, default='./dataset/')
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly')

parser.add_argument('--seq_len',    type=int, default=720)
parser.add_argument('--label_len',  type=int, default=48)
parser.add_argument('--pred_len',   type=int, required=True)
parser.add_argument('--period_len', type=int, required=True)
parser.add_argument('--enc_in',     type=int, required=True)
parser.add_argument('--dec_in',     type=int, default=7)
parser.add_argument('--c_out',      type=int, default=7)

# ---- TimeTucker structural (not searched, but read by the model) ----
parser.add_argument('--use_period_norm',   type=int,   default=1)
parser.add_argument('--use_revin',         type=int,   default=1)
parser.add_argument('--share_factors',     type=int,   default=1)
parser.add_argument('--use_orthogonal',    type=int,   default=1)
parser.add_argument('--orthogonal_weight', type=float, default=0.005)

# ---- optimization (most are FIXED during search) ----
parser.add_argument('--num_workers',   type=int,   default=10)
parser.add_argument('--clip',          type=float, default=1.0)
parser.add_argument('--train_epochs',  type=int,   default=10)
parser.add_argument('--patience',      type=int,   default=3)
parser.add_argument('--loss',          type=str,   default='mse')
parser.add_argument('--lradj',         type=str,   default='type3')
parser.add_argument('--pct_start',     type=float, default=0.3)
parser.add_argument('--use_amp',       action='store_true', default=False)
parser.add_argument('--embed',         type=str,   default='learned')

# ---- BO config ----
parser.add_argument('--init_points', type=int, default=8)
parser.add_argument('--n_iter',      type=int, default=25)
parser.add_argument('--result_dir',  type=str, default='./logs/Bayes')

# ---- post-search retrain ----
parser.add_argument('--retrain',          type=int, default=1,
                    help='1=retrain best config with full epoch budget after BO; 0=skip')
parser.add_argument('--retrain_epochs',   type=int, default=30)
parser.add_argument('--retrain_patience', type=int, default=5)
parser.add_argument('--des',              type=str, default='bayes')

# ---- misc ----
parser.add_argument('--seed',          type=int,    default=2026)
parser.add_argument('--use_gpu',       type=str2bool, default=True)
parser.add_argument('--gpu',           type=int,    default=0)
parser.add_argument('--use_multi_gpu', type=int,    default=0)
parser.add_argument('--devices',       type=str,    default='0,1')
parser.add_argument('--output_attention', action='store_true', default=False)

# Placeholders the model/data layer touches but search doesn't tune. Default
# values are overwritten per trial when relevant (r_c, r_n, batch_size, ...).
parser.add_argument('--r_c',           type=int,    default=7)
parser.add_argument('--r_n',           type=int,    default=6)
parser.add_argument('--r_p',           type=int,    default=8)
parser.add_argument('--dropout',       type=float,  default=0.1)
parser.add_argument('--learning_rate', type=float,  default=0.01)
parser.add_argument('--batch_size',    type=int,    default=128)

args = parser.parse_args()
args.use_gpu = bool(args.use_gpu and torch.cuda.is_available())

print('Args in experiment:')
print(args)

tuner = TimeTuckerBayesTuner(
    base_args=args,
    init_points=args.init_points,
    n_iter=args.n_iter,
    result_dir=args.result_dir,
    retrain=bool(args.retrain),
    retrain_epochs=args.retrain_epochs,
    retrain_patience=args.retrain_patience,
)
tuner.run()
