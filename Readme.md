# DeepERA: deep learning enables comprehensive identification of drug-target interactions via embedding of heterogeneous data
## Prerequists (tested version):
- Python (3.6.8)
- pandas (0.21.1)
- pytorch (1.2.0)
- rdkit (2019.09.3.0)
- scikit-learn (0.20.3)
- numpy (1.16.1)

We recommend installing the above packages with the tested version to avoid any potential errors/warnings due to version changes.

## Installation with conda environment:
1. Create environment with Python 3.6.8:
```
conda create -n DeepERA python=3.6.8
```
2. Install packages in the DeepERA environment
```
conda activate DeepERA
conda install -c conda-forge scikit-learn=0.20.3
conda install -c rdkit rdkit=2019.09.3.0
conda install pandas=0.24.1
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install numpy=1.16.1
```
The installation of pytorch depends on the cuda version. If your cuda version is 9.2, please use the following command instead:
```
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
```
Please refer to the [instruction page of pytorch installation](https://pytorch.org/get-started/previous-versions/) if non-linux platform (e.g. MacOS, Windows) is used. 

## Run the code
1. Process raw data and processed data are stored in the folder *Processed_data*. Please replace the files in the folder *Raw_data* to run your own task:
```
python preprocess_data_DeepERA.py
```
2. Run training and prediction for the data in the folder *Processed_data*:
```
bash run_DeepERA.sh
```

## Citation
Please cite the following paper:
- Le Li, Shayne D. Wierbowski, Haiyuan Yu. DeepERA: deep learning enables comprehensive identification of drug-target interactions via embedding of heterogeneous data. Manuscript in submission.

## Contact
Please contact me (ll863@cornell.edu) for any questions.
