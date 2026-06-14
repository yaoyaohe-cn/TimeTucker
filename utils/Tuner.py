import copy
import csv
import datetime
import importlib
import os
import random
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from exp.exp_main import Exp_Main

def _import_optuna():
    try:
        return importlib.import_module('optuna')
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError('Optuna is not installed. Run: pip install optuna') from e

optuna = _import_optuna()

class Tuner:
    def __init__(self, ranSeed, n_jobs):
        self.fixedSeed = ranSeed
        self.n_jobs = n_jobs
        if self.n_jobs > 1:
            print("=========================================================================")
            print("WARNING: n_jobs > 1 detected! PyTorch global seeds WILL be polluted")
            print("across Python threads. Trials may not be reproducible. For rigorous")
            print("benchmarking, n_jobs MUST be 1 per node. Rely on Bash Background workers instead.")
            print("=========================================================================")
        self.result_dic = defaultdict(list)
        self.current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    def optuna_objective(self, trial, args):
        trial_args = copy.deepcopy(args)
        params = self._suggest_timetucker_params(trial, args)

        for key, value in params.items():
            setattr(trial_args, key, value)

        # 【改造点 1：Trial-specific seed，探索初始化分布】
        trial_seed = self.fixedSeed + trial.number
        
        setting = '{}_{}_sl{}_pl{}_rn{}_rc{}_ow{}_lr{}_bs{}_trial{}_sd{}'.format(
            trial_args.model, trial_args.data, trial_args.seq_len, trial_args.pred_len,
            trial_args.r_n, trial_args.r_c, trial_args.orthogonal_weight,
            trial_args.learning_rate, trial_args.batch_size, trial.number, trial_seed
        )

        self._set_random_seed(trial_seed)
        exp = Exp_Main(trial_args)

        try:
            exp.train(setting, optunaTrialReport=trial)
            val_loss = exp.vali_from_setting(setting)
            return val_loss
        except optuna.exceptions.TrialPruned:
            raise
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print('Trial pruned due to CUDA OOM.')
                self._cleanup_cuda()
                raise optuna.exceptions.TrialPruned()
            else:
                traceback.print_exc()
                raise e
        finally:
            self._cleanup_cuda()

    def tune(self, args):
        if args.model != 'TimeTucker':
            raise ValueError('TimeTucker Optuna tuner only supports --model TimeTucker.')

        output_dir = getattr(args, 'optuna_results_dir', './Output')
        os.makedirs(output_dir, exist_ok=True)

        sampler = optuna.samplers.TPESampler(seed=self.fixedSeed)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=getattr(args, 'optuna_n_startup_trials', 8),
            n_warmup_steps=1,
            interval_steps=1,
        )
        
        # 使用 SQLite 允许多进程并发
        study_name = f"{args.model}_{args.data}_{args.pred_len}_study"
        db_path = os.path.join(output_dir, f"{study_name}.db")
        storage_url = f"sqlite:///{db_path}"

        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
        )

        self.study.optimize(
            lambda trial: self.optuna_objective(trial, args),
            n_trials=args.optuna_trial_num,
            n_jobs=self.n_jobs,
            gc_after_trial=True,
        )

        self._final_evaluation_with_best_params(args)
        self.save_result(args)

    def _suggest_timetucker_params(self, trial, args):
        params = {}
        
        if getattr(args, 'optuna_period_search', False):
            period_choices = self._split_int_choices(getattr(args, 'optuna_period_choices', '12,24,48'))
            period_len = trial.suggest_categorical('period_len', period_choices)
        else:
            period_len = args.period_len
        params['period_len'] = period_len

        # 动态锁死边界，防破坏数学机制
        max_rc = min(32, args.enc_in)
        if max_rc <= 4:
            params['r_c'] = trial.suggest_int('r_c', 1, max_rc)
        else:
            params['r_c'] = trial.suggest_int('r_c', 4, max_rc, step=4)

        seg_num_x = args.seq_len // period_len
        max_rn = min(12, seg_num_x)
        if max_rn < 2:
            params['r_n'] = 1 
        else:
            params['r_n'] = trial.suggest_int('r_n', 2, max_rn, step=2)

        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 5e-1, log=True)
        params['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128])
        params['orthogonal_weight'] = trial.suggest_float('orthogonal_weight', 0.0, 0.20, step=0.01)

        orthogonal_choices = self._split_int_choices(getattr(args, 'optuna_use_orthogonal_choices', '0,1'))
        if len(orthogonal_choices) == 1:
            params['use_orthogonal'] = orthogonal_choices[0]
        else:
            params['use_orthogonal'] = trial.suggest_categorical('use_orthogonal', orthogonal_choices)

        return params

    def _final_evaluation_with_best_params(self, args):
        print('\n' + '=' * 60)
        print('Phase 2: Final Robust Evaluation (Validation Selection)')
        print('=' * 60)

        final_args = copy.deepcopy(args)
        for key, value in self.study.best_params.items():
            setattr(final_args, key, value)
            
        if 'use_orthogonal' not in self.study.best_params:
            final_args.use_orthogonal = self._split_int_choices(getattr(args, 'optuna_use_orthogonal_choices', '1'))[0]

        # 【改造点 2：串行多 Seed 验证，严格选出最优模型】
        num_eval_seeds = 3  
        best_val_loss = float('inf')
        best_setting = None
        best_seed = None

        print(f"Retraining best params: {self.study.best_params}")

        for i in range(num_eval_seeds):
            current_seed = self.fixedSeed + i
            setting = '{}_{}_sl{}_pl{}_rn{}_rc{}_ow{}_lr{}_bs{}_final_sd{}'.format(
                final_args.model, final_args.data, final_args.seq_len, final_args.pred_len,
                final_args.r_n, final_args.r_c, final_args.orthogonal_weight,
                final_args.learning_rate, final_args.batch_size, current_seed
            )

            print(f"\n--- Retraining with Robust Seed {i+1}/{num_eval_seeds} (Seed: {current_seed}) ---")
            self._set_random_seed(current_seed)
            exp = Exp_Main(final_args)
            exp.train(setting, optunaTrialReport=None)
            
            val_loss = exp.vali_from_setting(setting)
            print(f">>> Seed {current_seed} Validation Loss: {val_loss:.7f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_seed = current_seed
                best_setting = setting

        print('\n' + '*' * 60)
        print(f'Selection Complete: Seed {best_seed} won with Val Loss {best_val_loss:.7f}')
        print('*' * 60)

        # 【改造点 3：对最终模型在 Test Set 进行一锤定音盲测】
        print('\nEvaluating on independent TEST set ONCE using the selected model...')
        
        exp = Exp_Main(final_args)
        test_result = exp.test(best_setting, test=1) 
        
        if test_result is None:
            test_mse, test_mae = np.nan, np.nan
        else:
            test_mse, test_mae = test_result

        self.result_dic['final_test_mse'].append(test_mse)
        self.result_dic['final_test_mae'].append(test_mae)

        print('=' * 60)
        print(f'Final Reportable Test MSE: {test_mse:.6f}')
        print(f'Final Reportable Test MAE: {test_mae:.6f}')
        print('=' * 60 + '\n')

    def save_result(self, args):
        file_name = '{}_{}_len{}'.format(args.model, args.data, args.pred_len)
        output_dir = getattr(args, 'optuna_results_dir', './Output')
        os.makedirs(output_dir, exist_ok=True)

        best_params = dict(self.study.best_params)
        self.result_dic['model'].append(args.model)
        self.result_dic['data'].append(args.data)
        self.result_dic['seq_len'].append(args.seq_len)
        self.result_dic['pred_len'].append(args.pred_len)
        self.result_dic['best_val_loss'].append(self.study.best_value)

        for key, value in best_params.items():
            self.result_dic[key].append(value)

        best_path = os.path.join(output_dir, f'{file_name}_best_{self.current_time}.csv')
        trials_path = os.path.join(output_dir, f'{file_name}_trials_{self.current_time}.csv')
        self._write_result_csv(best_path)
        self._write_trials_csv(trials_path)
        print(f'Optimization results saved to {best_path}')

    def _write_result_csv(self, path):
        fieldnames = list(self.result_dic.keys())
        row_count = len(next(iter(self.result_dic.values())))
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for index in range(row_count):
                writer.writerow({key: values[index] for key, values in self.result_dic.items()})

    def _write_trials_csv(self, path):
        param_names = sorted({key for trial in self.study.trials for key in trial.params.keys()})
        fieldnames = ['number', 'state', 'value'] + param_names
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trial in self.study.trials:
                row = {'number': trial.number, 'state': trial.state.name, 'value': trial.value}
                row.update(trial.params)
                writer.writerow(row)

    @staticmethod
    def _split_int_choices(value):
        return [int(item.strip()) for item in value.split(',') if item.strip()]

    @staticmethod
    def _set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _cleanup_cuda():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
