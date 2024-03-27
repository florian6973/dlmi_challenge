# Challenge DLMI 2024

Code for the contribution of Diffused Burgers to the Kaggle challenge *Classification of lymphocytosis from white blood cells*.

## Installation

- Install Python 3.11
- Install the package `pip install -e .`
- Unzip the raw data in `data/raw`

The main function is in `src/dlmi/__init__.py`

## Use

- Start mlflow: run the command `mlflow_run`
- Launch the program (training+inference on test): `dlmi_train`
    - You can specify the config file `dlmi_train <path/to/config_file.yaml>`: some examples are in `src/dlmi/conf`

Tested on Linux + Nvidia V100S 32GB. Training time of less than 30 minutes.

### Main models

- Simple MLP on clinical attributes
- Mixture Of Experts MLP on clinical attributes + CNN on images

### Results

See report.
