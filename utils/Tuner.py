def optuna_objective(self, trial, args):
        trial_args = copy.deepcopy(args)
        params = self._suggest_timetucker_params(trial, args)

        for key, value in params.items():
            setattr(trial_args, key, value)
        trial_args.basis_num = trial_args.r_n

        # 【改造点 1：Trial-specific seed】
        # 让每个 trial 拥有自己独立的随机种子，既探索超参，又探索初始化分布
        trial_seed = self.fixedSeed + trial.number
        
        setting = '{}_{}_sl{}_pl{}_rn{}_rc{}_ow{}_lr{}_bs{}_trial{}_sd{}'.format(
            trial_args.model, trial_args.data, trial_args.seq_len, trial_args.pred_len,
            trial_args.r_n, trial_args.r_c, trial_args.orthogonal_weight,
            trial_args.learning_rate, trial_args.batch_size, trial.number, trial_seed
        )

        # 应用该 Trial 的专属种子
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
                import traceback
                traceback.print_exc()
                raise e
        finally:
            self._cleanup_cuda()

    def _final_evaluation_with_best_params(self, args):
        print('\n' + '=' * 60)
        print('Phase 2: Final Robust Evaluation (Validation Selection)')
        print('=' * 60)

        final_args = copy.deepcopy(args)
        for key, value in self.study.best_params.items():
            setattr(final_args, key, value)
            
        if 'use_orthogonal' not in self.study.best_params:
            final_args.use_orthogonal = self._split_int_choices(getattr(args, 'optuna_use_orthogonal_choices', '1'))[0]
        final_args.basis_num = final_args.r_n

        # 【改造点 2：串行多 Seed 验证，选出最优模型】
        num_eval_seeds = 3  # 你可以定义最后用几个不同的 Seed 来复测
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
            
            # 正常训练
            exp.train(setting, optunaTrialReport=None)
            
            # 提取该 seed 下在验证集上的表现
            val_loss = exp.vali_from_setting(setting)
            print(f">>> Seed {current_seed} Validation Loss: {val_loss:.7f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_seed = current_seed
                best_setting = setting

        print('\n' + '*' * 60)
        print(f'Selection Complete: Seed {best_seed} won with Val Loss {best_val_loss:.7f}')
        print('*' * 60)

        # 【改造点 3：测试集一锤定音】
        print('\nEvaluating on independent TEST set ONCE using the selected model...')
        
        # 重新初始化 Exp_Main 并加载我们挑选出的 best_setting
        exp = Exp_Main(final_args)
        # test=1 参数会让底层 exp_main.py 直接加载 best_setting 的 checkpoint.pth 进行评估，不再训练
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
