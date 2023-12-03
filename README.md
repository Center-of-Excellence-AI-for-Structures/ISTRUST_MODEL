##  Interpretable Spatiotemporal TRansformer for Understanding STructures (ISTRUST)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Code for the paper "Breaking the black box barrier: predicting remaining useful life under uncertainty from raw images with interpretable neural networks". This is a novel interpretable transformer-based model for Remaining Useful Life (RUL) prediction from raw sequential images (frames) representing a composite structure under fatigue loads.

![alt text](https://github.com/panoskom/ISTRUST_MODEL/blob/main/Figs/general_concept.jpg)



## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Structure](#structure)
- [Example](#example)
- [Contributors](#contributors)

## Requirements

- Windows 10
- Either GPU or CPU. However, a GPU is highly recommended (Tested on Nvidia RTX4080 16GB GPU).
- The developed version of the code mainly depends on the following 'Python 3.9.12' packages.

  ```
  torch==1.13.1
  torchvision==0.14.1
  torchaudio==0.13.1
  numpy==1.23.4
  pandas==1.5.3
  pillow==9.4.0
  matplotlib==3.7.1
  scipy==1.10.1
  joblib==1.2.0
  keyboard==0.13.5
  tqdm==4.64.0
  umap-learn==0.5.3
  ```

## Installation
Create an Anaconda environment and install PyTorch. In the 3rd step, please select the correct Pytorch version that matches your CUDA version from https://pytorch.org/get-started/previous-versions/. Open an Anaconda terminal and run the following:

```
conda create -n istrust python=3.9.12
conda activate istrust
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 pillow==9.4.0 --extra-index-url https://download.pytorch.org/whl/cu116
```
This repository can be directly installed through GitHub by the following commands:

```
conda install git
git clone https://github.com/panoskom/ISTRUST_MODEL.git
cd ISTRUST_MODEL
python setup.py install
conda install numpy-base==1.23.4
```

## Structure

In this project, we use an experimental dataset that can be downloaded via the following DOI: 10.17632/ky3gb8rk9h.1 

In the link above, information can be found about installing and understanding the dataset. **The only required folder to run the code is the** `dataset` **folder**. Simply, **extract the** `dataset.zip` **inside the** `ISTRUST_MODEL/istrust_model` (required folder name `dataset`). The dataset folder should contain the extracted images representing the experimental data. 

The runs folder that already exists contains the trained models from the cross-validation process that are required to evaluate the results. If the user chooses to train the model from scratch this folder is not needed. However, for potential memory issues we do recommend using our already trained models instead.
 

## Example

To specifically describe how to train and use the ISTRUST model, we show an example below. To run the code from the Anaconda terminal with default values, go to the `istrust_model` folder inside the `ISTRUST_MODEL` directory and run the `main.py` file via the commands:

```
cd istrust_model
python main.py
```

If you want to change some of the default variables, for example, if the code has been already run once, some processes are not needed, therefore run the command:

`python main.py --create_data False --create_augmented_data False`

If you want to store the attention weights and the UMAP representations, and you don't want to create/augment the data again, run the command:

`python main.py --create_data False --create_augmented_data False --export_attention_weights True --export_umap True`

See the `main.py` file for different existing variables and options.

### Results

The results are saved inside the directory `../ISTRUST_MODEL/istrust/runs/`. The results concerning the RUL predictions under uncertainty and the interpretability of the ISTRUST model via spatiotemporal attention are shown below. The figure corresponds to 3 specimens; the two specimens to the left are classified as optimal results, while the last one corresponds to suboptimal results. Nevertheless, the reasons that separate a prediction into optimal/suboptimal are satisfactorily explained via this interpretability (see Results section in the paper).

![alt text](https://github.com/panoskom/ISTRUST_MODEL/blob/main/Figs/results.jpg)

## Contributors

[Panagiotis Komninos](https://github.com/panoskom)
[Aderik Verraest](https://github.com/aderikverraest)

