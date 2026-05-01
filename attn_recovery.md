# Attention Distribution Recovery

This file documents the attention distribution recovery methods, implemented on top of Kascade

The core idea of attention distribution recovery is that the distribution of the attention weight (result of $softmax(QK^T)$) changes as the pruning method forcely clamps the tiny attention values to zero.
Thus its successor components such as attention output ($AV$, and $Linear(AV)$) will also have different distributions compared to before pruning.
The sudden change on the distribution causes a mismatch between the expected mean and std of the LayerNorm's input, and potentially leads to accuracy drop.
The goal of attention distribution recovery is to shift the mean/std of the component so that it matches the LayerNorm's expectation.

## Terms

**Pruning**: in this project, pruning refers to Kascade pruning method.

**Attention component/components in attention:** refers to the intermediate results of attention mechanism including $Q, K, V, A=Softmax(QK^T), AV, O=Linear(AV)$

**recovery**: refers to the method of shifting the mean/std of a certain attention component after attention pruning, to recover the mean/std that are affected by the attention pruning

**data capture/attention component capture**: refers to saving the attention components to `.npy` files so that we can use them later for offline analysis and MLP training

**prediction**: refers to using an MLP to predict the mean and std of a certain attention component, so that we can use it to recover the mean/std of the component without computing the mean/std of the original component (without attention pruning)

**MLP-prediction-based recovery**: refers to the apporach of using a trained MLP to predict the mean and std of the attention component, and then shift the mean/std of the attention component to match the predicted mean and std. The mean and std is predicted because they are unknown with attention pruning.

**non-prediction-based recovery**: refers to the approach to recover the mean/std of the attention component by targeting at "ground-truth" mean/std, which is done in a 2-pass apporach:

- 1st pass: compute the mean/std of the original attention component (without pruning)

- 2nd pass: compute the mean/std of the attention component with attention pruning, and then shift the mean/std of the attention component to match the "ground-truth" mean/std

This apporach does not save any computation from sparse attention, it is just a way to verify if the attention distrubtion recovery could lead to accuracy improvement, assuming a "perfect" recovery could be achieved.

## What's in this repo

This repository is forked from Kascade, an importance-score-based pruning method, to verify the hypothesis of attention distribution recovery.
Currently this repo finishes the implementation of:

- Attention component capturing method and analysis, for baseline (without pruning) and EfficientKascadeStrategy (flash-attention-based pruning)
- MLP model for predicting attention component mean/std
- MLP model training script and hyper-parameter tuning script
- MLP-prediction-based recovery method
- Non-prediction-based recovery method

## Prediction-based recovery method

Let `attn_component` the component after pruning that needs mean/std shifting, `pred_mu` and `pred_std` the predicted mean and std of `attn_component` respectively.
The `pred_mu` and `pred_std` are predicted by a small MLP, `mlp_recovery_model`, with input `x`.
`x` can be some extracted features of attention's component, such as mean/std of self-attention input, L2-norm of attention input, etc.
The code to shift the mean/std of `attn_component` is as follows:

```python
# predict mu and sigma as shifting target
pred_mu, pred_std = mlp_recovery_model(x)
# figure out shifting amount shape
orignal_shape = attn_component.shape
pred_mu = pred_mu.view(*original_shape[:2], 1).to(attn_component.dtype)
pred_std = pred_std.view(*original_shape[:2], 1).to(attn_output.dtype)
# shifting to predicted mean and std
eps = 1e-8
curr_mu = attn_component.mean(dim=-1, keepdim=True)
curr_std = attn_component.std(dim=-1, keepdim=True)
attn_component.sub_(curr_mu).div_(curr_std + eps).mul_(pred_std).add_(curr_mu)
```

## List of changes in Kascade to support prediction-based recovery experiments

### [`src/attn_recovery/`](src/attn_recovery)
- Add [`RecoveryMLP`](src/attn_recovery/recovery_mlp.py) class for prediction-based mean/std shifting.

### [`src/strategies/`](src/strategies)
- Add [`verify_pruning_recovery_strategy.py`](src/strategies/verify_pruning_recovery_strategy.py) to run the non-prediction recovery method. This method is used to verify if a "perfect" recovery could help boost the accuracy.
- Add [`EfficientKascadeRecoveryStrategy class`](src/strategies/EfficientKascadeStrategy.py#L175) for Efficient Kascade pruning strategy with MLP-prediction-based attention recovery

### [`scripts/`](scripts/)

- Add [`recovery_mlp_train.py`](scripts/recovery_mlp_train.py) to train the prediction-based recovery MLP model.
- Add [`recovery_mlp_tune.py`](scripts/recovery_mlp_tune.py) to search the best hyper-parameters of recovery MLP, using optuna
- Add [`profile_and_eval_recovered_attnout_lb.py`](scripts/profile_and_eval_recovered_attnout_lb.py) to:
    - store activation such as attention input,  attention output and Value matrix on LongBench using baseline model (without pruning)
    - run recovery verification experiment (recover the mean/std without MLP prediction (just to verify if a "perfect" recovery could help boost the accuracy))
    - run MLP-prediction-based recovery experiment on LongBench (this is with trained MLP to predict mean/std)
- Modify [`eval_script.py`](scripts/eval_script.py) to support new strategies for different experiments
    - `baseline_profile` strategy for attention input/output/value matrix capturing
    - `efficient_kascade_recovery` strategy for MLP-prediction-based recovery
    - `verify_pruning_recovery` strategy for non-predicted (online computed) mean/std recovery
- Add [`stat_analyzer.py`](scripts/stat_analyzer.py) to extract and visualize the statistics of activations.

### [`src/model_utils.py`](src/model_utils.py)

- Add `get_attn_out_stat_profile()` to capture attention input/output and Value matrix from LongBench dataset eval
- Add `apply_mlp_recovery()` to insert MLP-prediction-based mean/std recovery into model evaluation.

## Usage

```bash
# if targeting at a model without anchor layer records, run this first to find the anchor layers
python scripts/eval_script.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name bdsaglam/musique \
    --subsets answerable \ 
    --num_queries 1000 \
    --strategies post_softmax_pooled_prefill_topk \ 
    --tile_size 32 \
    --run_type select_layers \ 

# evaluate models on longbench
python scripts/eval_lb.py

# profile model activation, the activation will be saved as npy files
python scripts/profile_and_eval_recovered_attnout_lb.py

# train recovery mlp using saved npy files, save parameters to model_path
# the trained model can be used by scripts/profile_and_eval_recovered_attnout_lb.py for MLP-prediction-based recovery
python scripts/recovery_mlp_train.py \
    --model_name "Qwen3-8B" \
    --data_base_path "./results/attn_recovery/train" \
    --model_path "./results/attn_recovery/mlp_model_vmatrix_layer_distinguished_stdonly"

# tune hyper parameter of mlp training using optuna
python scripts/recovery_mlp_tune.py \
    --model_name "Qwen3-8B" \
    --data_base_path "./results/attn_recovery/train" \
    --n_trials 20 \
    --tune_epochs 30 \
    --rep_layer "layer_5"

# analyze correlation between statistics of activations and attention inputs
python scripts/stat_analyzer.py \
  --model_name "Qwen3-8B" \
  --data_base_path "./results/attn_recovery/train" \
  --out_dir "./results/attn_recovery/corr_analysis"

```

## Preliminary results

#### 1. Non-prediction-based attention recovery does not significantly boost the accuracy.
    
We used 2-pass method to recover the weighted sum of values ($AV$), by shifting the mean and std of $AV$ after pruning, to the mean and std of $AV$ before pruning.
By comparing `results/evals/Qwen3-8B/LongBench/efficient_kascade_top5.csv` and `results/evals/Qwen3-8B/LongBench/verify_pruning_recovery_top5.csv`, there seems no sigificant accuracy improvement by attention recovery. 

#### 2. Predicting std using MLP has various effectiveness on different layers

`results/attn_recovery/mlp_model_layer_distinguished/` records the training log of MLP which predicts the std of $AV$ using self-attention input (hidden states).
The $R^2$ score is selected as the test metric of MLP model.
For different MLPs that are that are trained for different layers, the test $R^2$ score fluctuates. 

#### 3. Various level of correlations between statistics of attention inputs and $AV$'s mean and std

`results/attn_recovery/corr_analysis/` records the correlation analysis between statistics of activations and attention inputs.
For different layers, the correlation level varries. 
In some layers, multiple statistics of attention inputs are highly correlated with $AV$'s mean and std.
Others has no obvious correlation.

#### 4. Non-prediction-based attention recovery improves some tasks' accuracy when the recovery is performed on the $Softmax(QK^T)$

This experiment is performed in another [repo](https://github.com/COMPAS-Lab/sparsity-compas-forked-longbench/tree/flexattn).
On Olmo-2-1B model with element-wise, threshold based pruning with eager attention, the recovery of $A= Softmax(QK^T)$ improves the LongBench score on some specific tasks.

## Future plans

In the future the attention recovery research should include two separate research platforms: 
1. eager-attention-based platform, for maximum flexibility of designing recovery strategies
2. flash-attention-based platform, for recovery evaluation

### Eager-attention-based platform:

Eager-attention-based experiment platform should be built based on top of [`KascadeStrategy`](src/strategies/KascadeStrategy.py) which does not use flash-attention kernel to perform attention calculation.
The goal of this platform is to provide the ability to store and alter any possible node inside attention, including:
- the input of the attention layer
- Q, K, V matrices
- attention weights ($A=softmax(QK^T)$)
- weighted value sums ($AV$)
- output of attention layer ($O=Linear(AV)$)

So that we can figure out which component is best to be recovered, to minimize the accuracy drop.
The attention components altered and stored by the eager-attention-based platform can be used to 

- visualize the distribution and statistics of the attention components
- visualize the change before and after pruning
- quantitatively records the attention sparsity and LongBench scores
- integrate non-MLP prediction-based recovery to validate the effectiveness of recovery method, and select the best component to recover the statistics
- integrate MLP recovery to evaluate and validate the effectiveness of predicting the statistics of selected component

Eager-attention-based platform should have the restriction of running on a set of limited length sequences to prevent from OOM errors.
This restriction can be implemented in its strategies class, so that it does not need the interference from evaluation scripts (`scripts/eval_script.py` and `scripts/eval_lb.py`).

### Flash-attention-based platform:

The main goal of flash-attention-based platform is to evaluate the attention recovery method over the entire LongBench dataset. 
Once the recovery method is validated on the eager-attention-based platform, we should proceed to evaluate it on the flash-attention-based platform.
This platform should be built on top of [`EfficientKascadeStrategy`](src/strategies/EfficientKascadeStrategy.py), which uses flash-attention kernel to perform attention calculation.

If the component that is selected to recover is outside the Huggingface attention kernel (which has the input as $Q, K, V$, and output as $AV$), the current implementation in [`src/model_utils.py`](src/model_utils.py#L145) can be reused. 
Otherwise we need to consider a customized kernel that integrates attention recovery into flash-attention kernel. 

### Roadmap

1. Build the eager-attention-based platform
    - need to support different components to recover, including $A=softmax(QK^T)$, $AV$, and $O$
    - need to support kascade pruning strategies
    - need to support data capturing (saving $A=softmax(QK^T)$, $AV$, and $O$ to disk)
    - need to support both MLP-prediction-based and non-prediction-based recovery
    - need to support limited context length to prevent OOM errors

2. Verify and repreduce the result of $A=softmax(QK^T)$ recovery
    - compare the Longbench score of non-prediction-based $A=softmax(QK^T)$ recovery, with KascadeStrategy pruning.

3. Using eager-attention-based platform to figure out the best component to recover

4. Integrate prediction-based recovery into flash-attention-based platform, also consider the feasibility of recovering the best component (figured out in step 3) with flash-attention

5. Building MLP-prediction-based recovery (supposing the predictor takes X or Q/K/V as input, and outputs the mean and std of the selected component to recover)
    - Analyze the correlation between statistics of the predictor's input (X, Q, K, V, etc) and the mean and std of the component selected in step 3, to confirm if it is feasible to use MLP to predict the mean/std of the selected component
    - Train MLP using the captured X and the mean/std of the selected component (the capture is either done by the eager-attention-based platform or the flash-attention-based platform)
        - Consider using different loss function and test metrics such as $MSE$, $R^2$, $KL$ divergence, etc.
    - Verify the accuracy of pruned model with MLP-prediction-based recovery.        

