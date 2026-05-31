# Lipid Nanoparticle Design with Composite Material Transformer (COMET)
===================================================================

## Setup and Installation

### Requirements
- Python 3.10
- CUDA 11.6
- PyTorch 1.13.1
- Anaconda 23.1.0

### Creating the Environment
1. Load the required modules and create a new Anaconda environment:
    ```bash
    conda create -n comet_env python=3.10
    source activate comet_env
    ```

2. Install dependencies:
    ```bash
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
    pip install lmdb==1.4.0 ml-collections==0.1.1 numpy==1.23.4 scipy==1.9.3 tensorboardX==2.5.1 tqdm==4.64.1 tokenizers==0.13.2 pyprojroot==0.2.0 pandas==1.5.2 scikit-learn==1.2.0 rdkit-pypi==2022.9.3
    ```

    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
    pip install lmdb==1.4.0 ml-collections==0.1.1 numpy==1.23.4 scipy==1.9.3 tensorboardX==2.5.1 tqdm==4.64.1 tokenizers==0.13.2 pyprojroot==0.2.0 pandas==1.5.2 scikit-learn==1.2.0 rdkit-pypi==2022.9.3

3. Install Uni-Core, compatible with the specified versions:
    ```bash
    pip install https://github.com/dptech-corp/Uni-Core/releases/download/0.0.2/unicore-0.0.1+cu116torch1.13.1-cp310-cp310-linux_x86_64.whl
    ```

## Data Preprocessing
Preprocessing is done to make lmdb dataset (stored in `processed_data_dirs/`) from json files (stored in `data_json/`). Scripts for generating processed datasets from JSON files:
- `experiments/preprocess_data_LANCE.ipynb`: Processes data for LNPs' efficacy on DC2.4 and B16-F10 cells.
- `experiments/preprocess_data_CACO2.ipynb`: Includes CACO2 cell transfection data.
- `experiments/preprocess_data_stability.ipynb`: Processes data for lyophilized LNPs.

## Structured LNPDB Pipeline
The LNPDB heart/kidney workflow is configured in `configs/lnpdb_heartkidney.json` and wrapped by scripts in `scripts/lnpdb/`.

```bash
python -m scripts.lnpdb.prepare_data
python -m scripts.lnpdb.inspect_data --fold 0
python -m scripts.lnpdb.train --fold 0 --dry-run
python -m scripts.lnpdb.summarize_results
python -m scripts.lnpdb.summarize_logs --contains bs8
```

See `scripts/lnpdb/README.md` for the full prepare -> preprocess -> train -> infer -> summarize flow.

## Training
Training scripts for different models:
- Lipid-only LNP: `experiments/training_script_LANCE_lipid_T01.py`
- PBAE LNP: `experiments/training_script_LANCE_PBAE_T02.py`
- Different cell types (CACO2): `experiments/training_script_caco2_T03.py`
- Lyophilized LNP: `experiments/training_script_stability_T04.py`

## Inference
Run inference using pretrained models:
- Lipid-only LNP: `experiments/inference_script_LANCE_lipid_I01.py`
- PBAE LNP: `experiments/inference_script_LANCE_PBAE_I02.py`

Predictions are stored as .pkl files in the specified output directories. These files end with ".out.pkl" in their names. Inference results will be outputted at location: dataset_output_dir>model_name>output_filename 

## Key Files & Folders
- `experiments/`: Contains training and preprocessing scripts.
- `experiments/task_schemas/`: stores task schema files: dictionary file to specify key information of multiple datasets. The keys are names of the datasets, value is a dictionary containing tasks_schema_path, component_types_schema_path and np_prop_schema_path as keys
- `experiments/data_json/`: stores raw LNP data
- `experiments/processed_data_dirs/`: stores processed lmdb datasets for training and inference
- `experiments/weights/`: stores pretrained COMET weights, accompanying example inference scripts to use them are `inference_script_*.py` ([download example weights here](https://drive.google.com/drive/folders/1IBz8iWrPX5Xnlb02VaTNR-7xuKuYUHZv?usp=drive_link))
- `unimol/`: Core logic for model operations.
- `ckp/`: Pretrained model weights ([download here](https://drive.google.com/drive/folders/1Ul89o6Vj93T01foKa1-898H32bJLTJkm?usp=drive_link))

## Pretrained Weights
Available pretrained weights for different models are listed under `experiments/weights/`. These can be used directly for deploying models and running inference scripts.

### Note
Code has been test on Linux and installation is expected to take less than 2 hours on a typical computer with internet connection.
