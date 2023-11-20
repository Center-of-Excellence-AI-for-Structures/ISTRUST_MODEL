##  Interpretable Spatiotemporal TRansformer for Understanding STructures (ISTRUST)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Code for the paper "Breaking the black box barrier: predicting remaining useful life under uncertainty from raw images with interpretable neural networks". This is a novel interpretable transformer-based model for Remaining Useful Life (RUL) prediction from raw sequential images (frames) representing a composite structure under fatigue loads.

![alt text](https://github.com/panoskom/ISTRUST_MODEL/blob/main/Figs/general_concept.jpg)



## Table of Contents

- [Environment and Requirements](#requirements)
- [Configuration and Installation](#installation)
- [Data Structure](#structure)
- [Example](#example)

## Environment and Requirements

- Windows 10
- Either GPU or CPU. However, a GPU is highly recommended (Tested on Nvidia RTX4080 16GB GPU).
- The developed version of the code mainly depends on the following 'Python 3.9.12' packages.

  ```
  torch==1.11.0
  torchvision==0.15.2
  torchaudio==0.11.0
  numpy==1.23.4
  pandas==1.5.3
  pillow==9.4.0
  matplotlib==3.5.2
  scipy==1.10.1
  joblib==1.2.0
  keyboard==0.13.5
  tqdm==4.64.0
  umap-learn==0.5.3
  ```

## Environment Configuration and Installation
Create an Anaconda environment and install PyTorch. In the 3rd step, please select the correct Pytorch version that matches your CUDA version from https://pytorch.org/get-started/previous-versions/. Open an Anaconda terminal and run the following:

```
conda create -n istrust_model python=3.9.12
conda activate istrust_model
conda install pytorch==1.11.0 torchvision==0.15.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
This repository can be directly installed through GitHub by the following commands:

```
conda install git
git clone https://github.com/panoskom/ISTRUST_MODEL.git
cd ISTRUST_MODEL
python setup.py install
conda install numpy-base==1.23.4
```

## Data Structure

In this project, we use an experimental dataset that can be downloaded via the following DOI: 10.17632/ky3gb8rk9h.1 

In the link above, information about installing and understanding the dataset can be found. The 2 required folders that are needed to run the code are the dataset and the runs folder. The dataset folder should contain the extracted images representing the experimental data. The runs folder contains the trained models from the cross-validation process that are required to evaluate the results. If the user chooses to train the model from scratch this folder is not needed.
 

## Example

To specifically describe how to train and use the ISTRUST model, we show an example below. To run the code from the Anaconda terminal with default values, go to the `istrust` folder inside the `ISTRUST_MODEL` directory and run the `main.py` file via the commands:

```
cd istrust
python main.py
```

If you want to change some of the default variables, for example, ..., run the command:

`python main.py --bayesian_opt True --mimic True`

See the `main.py` file for different existing variables and options.

### Results

The results are saved inside the directory `../ISTRUST_MODEL/istrust/runs/`. The results concerning the RUL predictions under uncertainty and the interpretability of the ISTRUST model via spatiotemporal attention are shown below. The figure corresponds to 3 specimens; the two specimens to the left are classified as optimal results, while the last one corresponds to suboptimal results. Nevertheless, the reasons that separate a prediction into optimal/suboptimal are satisfactorily explained via this interpretability (see Results section in the paper).

![alt text](https://github.com/panoskom/ISTRUST_MODEL/blob/main/Figs/results.jpg)

