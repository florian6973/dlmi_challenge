# Challenge DLMI 2024

Code for the contribution of Diffused Burgers to the Kaggle challenge *Classification of lymphocytosis from white blood cells* (see https://www.kaggle.com/competitions/dlmi-lymphocytosis-classification).

## Installation

- Install Python 3.11
- Install the package `pip install -e .`
- Unzip the raw data in `data/raw`

The main function is in `src/dlmi/__init__.py`

## Use

- Start mlflow: run the command `mlflow_run` (you can access the web app at `http://localhost:5001/`)
- Launch the program (training+inference on test) from the main folder of the project: `dlmi_train`
    - You can specify the config file `dlmi_train --config-name [train_mlp,train_moe]`: some examples are in `src/dlmi/conf`
- Results are in the `outputs` folder, the prediction for the test set is in `submission_test_final.csv`

Tested on Windows + Nvidia RTX 4080 12GB and Linux + Nvidia V100S 32GB. Training time of less than 30 minutes on V100S.

### Main models

- Simple MLP on clinical attributes
- Mixture Of Experts MLP on clinical attributes + CNN on images

### Results

See report.

## Troubleshooting

- `MiniDataset` has not been updated to the latest version of the code (with Stratified K-Fold). Please use only `MILDataset` in the configs.
- `ResNet`` is the default backbone in the code. Switching to `ViT` can be done by uncommenting code in `src/dlmi/models/moe_model.py`. `ViT` was more computationnally-intensive and did not provide better results, hence our choice.
