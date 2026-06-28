"""
TimeTucker Bayesian hyper-parameter tuner.

Searches: r_c, r_n, r_p, learning_rate, batch_size, dropout, orthogonal_weight
Per-dataset bounds are defined in DATASET_SPACES below.

Continuous BO over [0, 1] for every variable, then mapped to the discrete
grid each variable lives on. This keeps the surrogate well-behaved and lets
us mix categorical (batch_size), stepped-int (r_c, r_n, r_p), stepped-float
(dropout) and log-scale-float (learning_rate, orthogonal_weight) cleanly.
"""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from bayes_opt import BayesianOptimization
from torch import optim
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_main import Exp_Main
from models import TimeTucker
from utils.tools import EarlyStopping, adjust_learning_rate


# ---------------------------------------------------------------------------
# Per-dataset search ranges (closed intervals, inclusive of both ends).
# Each entry: (low, high, step).
# r_p ranges are tuned to each dataset's period_len; verify they remain valid
# if you change period_len in the shell scripts.
# ---------------------------------------------------------------------------
DATASET_SPACES = {
    'ETTh1':       {'r_c': (2, 7,   1), 'r_n': (2, 12,  2), 'r_p': (4, 24, 2)},
    'ETTh2':       {'r_c': (2, 7,   1), 'r_n': (2, 12,  2), 'r_p': (4, 24, 2)},
    'ETTm1':       {'r_c': (2, 7,   1), 'r_n': (8, 64,  4), 'r_p': (2, 4,  1)},
    'ETTm2':       {'r_c': (2, 7,   1), 'r_n': (8, 64,  4), 'r_p': (2, 4,  1)},
    'Weather':     {'r_c': (4, 21,  2), 'r_n': (8, 64,  4), 'r_p': (2, 4,  1)},
    'Electricity': {'r_c': (16, 96, 8), 'r_n': (4, 12,  2), 'r_p': (4, 24, 2)},
    'Traffic':     {'r_c': (16, 128,8), 'r_n': (4, 12,  2), 'r_p': (4, 24, 2)},
}

# Shared (dataset-independent) ranges
LR_RANGE       = (1e-4, 5e-2)            # log-uniform
DROPOUT_RANGE  = (0.1, 0.5, 0.1)         # step 0.1 -> { 0.1, 0.2, 0.3, 0.4, 0.5}
BATCH_CHOICES  = [64, 128, 256]
ORTHO_RANGE    = (5e-3, 2e-1)            # log-uniform; ow=0 effectively means use_orthogonal=0


# ---------------------------------------------------------------------------
# Mapping helpers: continuous [0, 1] -> concrete value on the dataset's grid.
# Snap to the nearest grid point so the suggested config is reproducible.
# ---------------------------------------------------------------------------
def _snap_int(u: float, low: int, high: int, step: int) -> int:
    """Map u in [0,1] to the integer grid {low, low+step, ..., <=high}."""
    grid = list(range(low, high + 1, step))
    idx  = min(int(u * len(grid)), len(grid) - 1)
    return grid[idx]


def _snap_float(u: float, low: float, high: float, step: float) -> float:
    grid = []
    v = low
    while v <= high + 1e-9:
        grid.append(round(v, 6))
        v += step
    idx = min(int(u * len(grid)), len(grid) - 1)
    return grid[idx]


def _log_uniform(u: float, low: float, high: float) -> float:
    log_low, log_high = np.log(low), np.log(high)
    return float(np.exp(log_low + u * (log_high - log_low)))


def _snap_choice(u: float, choices):
    idx = min(int(u * len(choices)), len(choices) - 1)
    return choices[idx]


def decode_params(data_name, u_rc, u_rn, u_rp, u_lr, u_bs, u_drop, u_ow):
    """Decode 7 continuous [0,1] BO suggestions into concrete hyper-params."""
    space = DATASET_SPACES[data_name]
    rc_lo, rc_hi, rc_step = space['r_c']
    rn_lo, rn_hi, rn_step = space['r_n']
    rp_lo, rp_hi, rp_step = space['r_p']
    return {
        'r_c':               _snap_int(u_rc, rc_lo, rc_hi, rc_step),
        'r_n':               _snap_int(u_rn, rn_lo, rn_hi, rn_step),
        'r_p':               _snap_int(u_rp, rp_lo, rp_hi, rp_step),
        'learning_rate':     _log_uniform(u_lr, *LR_RANGE),
        'batch_size':        _snap_choice(u_bs, BATCH_CHOICES),
        'dropout':           _snap_float(u_drop, *DROPOUT_RANGE),
        'orthogonal_weight': _log_uniform(u_ow, *ORTHO_RANGE),
    }


# ---------------------------------------------------------------------------
# Single trial: train `args.train_epochs` epochs and return best val MSE.
# Mirrors exp_main.train but strips test-set logging and checkpoint I/O so we
# can run many trials cheaply.
# ---------------------------------------------------------------------------
def _build_model(args, device):
    model = TimeTucker.Model(args).float().to(device)
    return model


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_one_trial(args, device) -> float:
    """Return best validation MSE seen across `args.train_epochs` epochs."""
    args.num_workers = 0
    _seed_everything(args.seed)

    train_data, train_loader = data_provider(args, flag='train')
    vali_data,  vali_loader  = data_provider(args, flag='val')

    model = _build_model(args, device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss() if args.loss == 'mse' else (
                nn.L1Loss() if args.loss == 'mae' else nn.SmoothL1Loss())

    scheduler = None
    if args.lradj == 'TST':
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=len(train_loader),
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate,
        )

    best_val = float('inf')
    patience_counter = 0

    for epoch in range(args.train_epochs):
        model.train()
        for batch in train_loader:
            batch_x = batch[0].float().to(device, non_blocking=True)
            batch_y = batch[1].float().to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs, ortho = model(batch_x)
            f_dim   = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            target  = batch_y[:, -args.pred_len:, f_dim:]
            loss    = criterion(outputs, target)
            back    = loss + args.orthogonal_weight * ortho
            back.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            if args.lradj == 'TST' and scheduler is not None:
                scheduler.step()

        # ----- validation -----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in vali_loader:
                bx = batch[0].float().to(device, non_blocking=True)
                by = batch[1].float()
                out, _ = model(bx)
                f_dim = -1 if args.features == 'MS' else 0
                out = out[:, -args.pred_len:, f_dim:].detach().cpu()
                by  = by[:, -args.pred_len:, f_dim:]
                val_losses.append(criterion(out, by).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

        if args.lradj != 'TST':
            adjust_learning_rate(optimizer, scheduler, epoch + 1, args, printout=False)

    # free GPU memory between trials
    del model, optimizer, train_loader, vali_loader, train_data, vali_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val


# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------
class TimeTuckerBayesTuner:
    def __init__(self, base_args: argparse.Namespace,
                 init_points: int = 8, n_iter: int = 25,
                 result_dir: str = './logs/Bayes',
                 retrain: bool = False, retrain_epochs: int = 30,
                 retrain_patience: int = 5):
        if base_args.data not in DATASET_SPACES:
            raise ValueError(f"No search space defined for data={base_args.data}. "
                             f"Known: {list(DATASET_SPACES)}")
        self.base_args        = base_args
        self.init_points      = init_points
        self.n_iter           = n_iter
        self.result_dir       = result_dir
        self.retrain          = retrain
        self.retrain_epochs   = retrain_epochs
        self.retrain_patience = retrain_patience
        os.makedirs(self.result_dir, exist_ok=True)

        self.device = torch.device(
            f'cuda:{base_args.gpu}' if (base_args.use_gpu and torch.cuda.is_available())
            else 'cpu'
        )
        self.trial_log = []

    # BayesianOptimization expects a function whose kwargs are the BO variables
    def _objective(self, u_rc, u_rn, u_rp, u_lr, u_bs, u_drop, u_ow):
        params = decode_params(self.base_args.data,
                               u_rc, u_rn, u_rp, u_lr, u_bs, u_drop, u_ow)
        trial_args = copy.deepcopy(self.base_args)
        for k, v in params.items():
            setattr(trial_args, k, v)

        t0 = time.time()
        try:
            val_mse = _train_one_trial(trial_args, self.device)
        except Exception as e:
            print(f"[trial FAILED] params={params}  err={e}")
            val_mse = 1e9  # heavy penalty so BO avoids this region
        dt = time.time() - t0

        record = {**params, 'val_mse': val_mse, 'seconds': round(dt, 1)}
        self.trial_log.append(record)
        print(f"[trial {len(self.trial_log):>3}] "
              f"r_c={params['r_c']:>3}  r_n={params['r_n']:>3}  "
              f"r_p={params['r_p']:>3}  lr={params['learning_rate']:.5f}  "
              f"bs={params['batch_size']:>3}  drop={params['dropout']:.2f}  "
              f"ow={params['orthogonal_weight']:.4f}  "
              f"-> val_mse={val_mse:.6f}  ({dt:.1f}s)")

        # BO maximizes -> return negative loss
        return -val_mse

    def run(self):
        pbounds = {
            'u_rc':   (0.0, 1.0),
            'u_rn':   (0.0, 1.0),
            'u_rp':   (0.0, 1.0),
            'u_lr':   (0.0, 1.0),
            'u_bs':   (0.0, 1.0),
            'u_drop': (0.0, 1.0),
            'u_ow':   (0.0, 1.0),
        }
        bo = BayesianOptimization(
            f=self._objective,
            pbounds=pbounds,
            random_state=self.base_args.seed,
            verbose=0,
        )
        bo.maximize(init_points=self.init_points, n_iter=self.n_iter)

        best = bo.max
        best_decoded = decode_params(self.base_args.data, **best['params'])
        best_decoded['val_mse'] = -best['target']

        tag = (f"{self.base_args.data}_sl{self.base_args.seq_len}_"
               f"pl{self.base_args.pred_len}_pp{self.base_args.period_len}")
        out_path = os.path.join(self.result_dir, f"{tag}.json")
        payload = {
            'data':         self.base_args.data,
            'seq_len':      self.base_args.seq_len,
            'pred_len':     self.base_args.pred_len,
            'period_len':   self.base_args.period_len,
            'init_points':  self.init_points,
            'n_iter':       self.n_iter,
            'search_epochs':   self.base_args.train_epochs,
            'search_patience': self.base_args.patience,
            'best':         best_decoded,
            'trials':       self.trial_log,
        }
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2)

        print(f"\n=== Best for {tag} ===")
        print(json.dumps(best_decoded, indent=2))
        print(f"Saved -> {out_path}")

        if self.retrain:
            print(f"\n=== Retraining best config for {self.retrain_epochs} epochs "
                  f"(patience={self.retrain_patience}) ===")
            retrain_result = self._retrain_best(best_decoded, tag)
            payload['retrain'] = retrain_result
            with open(out_path, 'w') as f:
                json.dump(payload, f, indent=2)
            print(f"Retrain result -> {retrain_result}")
            print(f"Updated  -> {out_path}")

        return best_decoded

    # -----------------------------------------------------------------------
    # Retrain best config with full epoch budget via the standard Exp_Main
    # pipeline so we get the same train/val/test metrics format as run_longExp.
    # -----------------------------------------------------------------------
    def _retrain_best(self, best_decoded: dict, tag: str) -> dict:
        retrain_args = copy.deepcopy(self.base_args)
        for k in ('r_c', 'r_n', 'r_p', 'learning_rate', 'batch_size',
                  'dropout', 'orthogonal_weight'):
            setattr(retrain_args, k, best_decoded[k])
        retrain_args.train_epochs = self.retrain_epochs
        retrain_args.patience     = self.retrain_patience
        retrain_args.model_id     = f"bayes_{tag}"
        retrain_args.des          = 'bayes_best'
        retrain_args.itr          = 1

        _seed_everything(retrain_args.seed)
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_{}_{}_seed{}'.format(
            retrain_args.model_id, retrain_args.model, retrain_args.data,
            retrain_args.features, retrain_args.seq_len, retrain_args.pred_len,
            retrain_args.des, 0, retrain_args.seed)

        exp = Exp_Main(retrain_args)
        exp.train(setting)
        mse, mae = exp.test(setting)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            'setting':        setting,
            'epochs':         self.retrain_epochs,
            'patience':       self.retrain_patience,
            'test_mse':       float(mse),
            'test_mae':       float(mae),
        }
