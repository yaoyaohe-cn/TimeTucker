import copy
import csv
import datetime
import importlib
import os
import random
import traceback
from collections import defaultdict

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
            print("benchmarking, n_jobs MUST be 1 per node. Rely on Bash background workers instead.")
            print("=========================================================================")
        self.result_dic = defaultdict(list)
        self.current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Optuna Objective
    def optuna_objective(self, trial, args):
        trial_args = copy.deepcopy(args)
        params = self._suggest_timetucker_params(trial, args)

        for key, value in params.items():
            setattr(trial_args, key, value)

        # Trial-specific seed，扩大初始化分布的覆盖面
        trial_seed = self.fixedSeed + trial.number

        setting = self._build_setting(trial_args, tag=f'trial{trial.number}_sd{trial_seed}')

        self._set_random_seed(trial_seed)
        exp = Exp_Main(trial_args)

        try:
            exp.train(setting, optunaTrialReport=trial)
            val_loss = exp.vali_from_setting(setting)
            return val_loss # 验证集损失
        except optuna.exceptions.TrialPruned:
            raise
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print('Trial pruned due to CUDA OOM.')
                self._cleanup_cuda()
                raise optuna.exceptions.TrialPruned()
            traceback.print_exc()
            raise
        finally:
            self._cleanup_cuda()

    # Study Driver
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

        is_master = os.environ.get('IS_OPTUNA_MASTER', '1') == '1'

        if args.optuna_trial_num > 0:
            self.study.optimize(
                lambda trial: self.optuna_objective(trial, args),
                n_trials=args.optuna_trial_num,
                n_jobs=self.n_jobs,
                gc_after_trial=True,
            )
        else:
            print('No Optuna search trials requested; loading existing study only.')

        if not is_master:
            print('IS_OPTUNA_MASTER=0: worker search complete; skipping final evaluation.')
            return

        self._final_evaluation_with_best_params(args)
        self.save_result(args)

    # Search Space — TimeTucker
    def _suggest_timetucker_params(self, trial, args):
        params = {}

        # 固定几何维度
        period_len = args.period_len
        seg_num_x = max(1, args.seq_len // period_len)

        # ---------- 1) Tucker Ranks (动态钳制，防止非满秩 NaN) ----------
        # r_c : 通道秩
        max_rc = min(32, args.enc_in)
        if max_rc <= 8:
            params['r_c'] = trial.suggest_int('r_c', 1, max_rc)
        else:
            params['r_c'] = trial.suggest_int('r_c', 8, max_rc, step=2)

        # r_n : 段(时间)秩
        max_rn = max(6, min(48, seg_num_x // 2))

        if max_rn <= 30:
            params['r_n'] = trial.suggest_int('r_n', 6, max_rn, step=3)
        else:
            params['r_n'] = trial.suggest_int('r_n', 24, max_rn, step=3)

        # r_p :
        max_rp = min(16, period_len)
        if max_rp <= 4:
            params['r_p'] = trial.suggest_int('r_p', 2, max_rp)
        else:
            params['r_p'] = trial.suggest_int('r_p', 4, max_rp, step=2)

        # ---------- 2) 优化器 ----------
        params['learning_rate'] = trial.suggest_float('learning_rate', 5e-4, 5e-2, log=True)
        params['batch_size'] = trial.suggest_categorical('batch_size', [64, 128, 256])

        # ---------- 3) 正则化 ----------
        params['orthogonal_weight'] = trial.suggest_float('orthogonal_weight', 1e-4, 5e-2, log=True)
        params['dropout'] = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)

        return params


    # Phase 2: Robust Retrain + Test
    def _final_evaluation_with_best_params(self, args):
        print('\n' + '=' * 60)
        print('Phase 2: Final Robust Evaluation (Validation Selection)')
        print('=' * 60)

        final_args = copy.deepcopy(args)
        for key, value in self.study.best_params.items():
            setattr(final_args, key, value)

        num_eval_seeds = 3
        best_val_loss = float('inf')
        best_setting = None
        best_seed = None

        print(f"Retraining best params: {self.study.best_params}")
        print(f"Fixed Architecture Flags -> RevIN: {final_args.use_revin}, Share Factors: {final_args.share_factors}, Ortho: {final_args.use_orthogonal}, Period Norm: {final_args.use_period_norm}")

        for i in range(num_eval_seeds):
            current_seed = self.fixedSeed + i
            setting = self._build_setting(final_args, tag=f'final_sd{current_seed}')

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

        # 最终在 Test Set 上一锤定音
        print('\nEvaluating on independent TEST set ONCE using the selected model...')
        exp = Exp_Main(final_args)
        test_result = exp.test(best_setting, test=1)

        if test_result is None:
            test_mse, test_mae = np.nan, np.nan
        else:
            test_mse, test_mae = test_result

        self.result_dic['final_test_mse'].append(test_mse)
        self.result_dic['final_test_mae'].append(test_mae)
        self.result_dic['final_val_loss'].append(best_val_loss)
        self.result_dic['final_seed'].append(best_seed)

        print('=' * 60)
        print(f'Final Reportable Test MSE: {test_mse:.6f}')
        print(f'Final Reportable Test MAE: {test_mae:.6f}')
        print('=' * 60 + '\n')

    # Persistence
    def save_result(self, args):
        file_name = f'{args.model}_{args.data}_len{args.pred_len}'
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

        # 列长对齐（防止部分键只 append 一次而其他键多次时 csv 越界）
        max_len = max(len(v) for v in self.result_dic.values())
        for k, v in self.result_dic.items():
            if len(v) < max_len:
                v.extend([''] * (max_len - len(v)))

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

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_setting(a, tag):
        """统一 checkpoint 命名，避免不同 trial 互相覆盖 checkpoint.pth。"""
        return (
            f'{a.model}_{a.data}_sl{a.seq_len}_pl{a.pred_len}'
            f'_rn{a.r_n}_rc{a.r_c}_rp{getattr(a, "r_p", "NA")}'
            f'_rev{int(getattr(a, "use_revin", 0))}'
            f'_sf{int(getattr(a, "share_factors", 0))}'
            f'_pn{int(getattr(a, "use_period_norm", 0))}'
            f'_ow{a.orthogonal_weight:.5f}_do{getattr(a, "dropout", 0.0)}'
            f'_lr{a.learning_rate:.5f}_bs{a.batch_size}_{tag}'
        )

    @staticmethod
    def _split_int_choices(value):
        return [int(item.strip()) for item in str(value).split(',') if str(item).strip() != '']

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
