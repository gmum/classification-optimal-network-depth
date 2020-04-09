# Code for paper: "Finding the Optimal Network Depth in Classification Tasks"

To run the code:
 - Edit the experiment.py main() function to select the setup (dataset,  hyperparameters etc.);
 - Run the experiment (example): 
```conda activate cuda100  
mkdir -p depth-experiment-results  
cd depth-experiment-results
python -u /path/to/code/experiment.py --dataset_dir /path/to/datasets/
```
- Most Figures from the paper can be generated with scripts in `tools`, for example:
```
python /path/to/code/tools/baseline_bar_plot.py n_runs_cifar_conv_baselines_5_and_betas --add n_runs_cifar_conv_baselines_5_and_betas/*@state
python /path/to/code/tools/importance_weights_plot.py --method brightness --batches_per_epoch 390 n_runs_cifar_conv_baselines_5_and_betas
```
