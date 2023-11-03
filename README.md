# ISTRUST_model

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Code for the paper "Breaking the black box barrier: predicting remaining useful life under uncertainty from raw images with interpretable neural networks". This is a novel interpretable transformer-based model for Remaining Useful Life (RUL) prediction from raw sequential images (frames) representing a composite structure under fatigue loads.

![alt text](https://github.com/panoskom/ISTRUST_model/blob/main/Figs/DC_model.jpg)


## Table of Contents

- [Environment and Requirements](#requirements)
- [Configuration and Installation](#installation)
- [Data Structure](#structure)
- [Example](#example)

## Environment and Requirements

- Windows 10
- Either GPU or CPU (Tested on Nvidia GeForce RTX 2080 GPU)
- The developed version of the code mainly depends on the following 'Python 3.9.12' packages.

  ```
  torch==1.11.0
  torchvision==0.12.0
  torchaudio==0.11.0
  numpy==1.23.4
  pandas==1.5.3
  matplotlib==3.5.2
  seaborn==0.12.1
  scikit-learn==1.2.2
  scikit-survival==0.21.0
  hmmlearn==0.3.0
  joblib==1.2.0
  tslearn==0.5.2
  tqdm==4.64.0
  ```

## Environment Configuration and Installation
Create an Anaconda environment and install PyTorch. In the 3rd step, please select the correct Pytorch version that matches your CUDA version from https://pytorch.org/get-started/previous-versions/. Open an Anaconda terminal and run the following:

```
conda create -n mono_dc python=3.9.12
conda activate mono_dc
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
This repository can be directly installed through github by the following commands:

```
conda install git
git clone https://github.com/panoskom/Monotonic_DC.git
cd Monotonic_DC
python setup.py install
conda install numpy-base==1.23.4
```

## Data Structure

In this project, two datasets are utilized, namely the MIMIC-III (https://mimic.mit.edu/) and the C-MAPSS dataset (https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6). 
The C-MAPSS dataset is publicly available and free. The only required file to be saved in the `CMAPS/` folder is the `train_FD001.txt` downloaded from the CMAPSS dataset.

The MIMIC-III dataset is publicly available and free, but it requires signing a data use agreement and passing a recognized course in protecting human research participants that includes Health Insurance Portability and Accountability Act (HIPAA) requirements. Approval requires at least a week.

After approval, the following CSV files should be saved in folder `MIMIC/data/`:

- ADMISSIONS.csv
- CHARTEVENTS.csv
- D_ITEMS.csv
- D_LABITEMS.csv
- LABEVENTS.csv
- PATIENTS.csv
 
### Data Files Distribution
The files and folders of the project are distributed in the following manner ('--Required' means that these files and folders are necessary to be created before running the `main.py`, the rest are automatically created)

```

../Monotonic_DC/
      └── setup.py
      └── Readme.md
      └── requirements.txt
      └── LICENSE
    
      ├── monotonic_dc/

            ├── bayesian_opt/                                -- Required      
            │ │   └── __init__.py                            -- Required 
            │ │   └── bayesian_optimization.py               -- Required 
            │ │   └── event.py                               -- Required 
            │ │   └── logger.py                              -- Required 
            │ │   └── observer.py                            -- Required 
            │ │   └── target_space.py                        -- Required 
            │ │   └── util.py                                -- Required 
            
            ├── CMAPS/                                       -- Required
            │ │   └── original/
            │ │ │    └── sp_0.csv
            │ │ │    └── sp_1.csv
            │ │ │    └── ...
            │ │   └── sorted/
            │ │ │    └── sp_0.csv
            │ │ │    └── sp_1.csv
            │ │ │    └── ...
            │ │   └── train_FD001.txt                        -- Required
            │ │   └── RUL_FD001.txt                          -- Required
            
            ├── events/
            │ │   └── test_cmaps_events.csv
            │ │   └── test_mimic_events.csv
            │ │   └── train_cmaps_events.csv
            │ │   └── train_mimic_events.csv
            
            ├── hyperparameters/
            │ │   └── hyper_cmaps.json
            │ │   └── hyper_mimic.json
            
            ├── MIMIC/                                        -- Required 
            │ │   └── data/                                   -- Required 
            │ │ │      └── ADMISSIONS.csv                     -- Required 
            │ │ │      └── CHARTEVENTS.csv                    -- Required 
            │ │ │      └── D_ITEMS.csv                        -- Required 
            │ │ │      └── D_LABITEMS.csv                     -- Required 
            │ │ │      └── LABEVENTS.csv                      -- Required 
            │ │ │      └── PATIENTS.csv                       -- Required 
            │ │   └── vital_signs/
            │ │ │      └── time_series_0.csv
            │ │ │      └── time_series_1.csv
            │ │ │      └── ...
            │ │   └── demographic.csv
            │ │   └── lab.csv
        
            
            ├── models/
            │ │   └── cmaps/
            │ │   └── mimic/
            │ │ │      └── ae_model_cmaps.pt
            │ │ │      └── dc_model_cmaps.pt
            
            ├── results/
            │ │   └── cmaps/
            │ │ │      └── cluster_embds/
            │ │ │      └── clusters/
            │ │ │      └── compare/
            │ │ │ |        └── clusters/
            │ │ │ |        └── prognostics/
            │ │ │      └── figs/
            │ │ │      └── loss/
            │ │ │      └── prognostics/
            │ │ │      └── time_grads/
            │ │ │      └── z_space
            │ │   └── mimic/
            │ │ │      └── cluster_embds/
            │ │ │      └── clusters/
            │ │ │      └── figs/
            │ │ │      └── loss/
            │ │ │      └── prognostics/
            │ │ │      └── time_grads/
            │ │ │      └── z_space

            ├── run_prognostics/                             -- Required      
            │ │   └── __init__.py                            -- Required 
            │ │   └── prognostic_models.py                   -- Required 

            ├── scalers/
            │ │   └── cmaps/
            │ │ │      └── scaler_t_f.save
            │ │ │      └── scaler_x.save
            │ │ │      └── scaler_y.save
            │ │   └── mimic/
            │ │ │      └── scaler_demo.save
            │ │ │      └── scaler_t_f.save
            │ │ │      └── scaler_x.save
            │ │ │      └── scaler_y.save
            
            ├── hyperparameters.py                           -- Required 
            ├── main.py                                      -- Required    
            ├── models.py                                    -- Required 
            ├── read_files.py                                -- Required 
            ├── run_models.py                                -- Required 
            ├── settings.py                                  -- Required 
            ├── utils.py                                     -- Required 
            ├── visualize.py                                 -- Required 

```

## Example

To specifically describe how to train and use the DC model, we show an example below. To run the code from the Anaconda terminal with default values, go to the `monotonic_dc` folder inside the `Monotonic_DC` directory and run the `main.py` file via the commands:

```
cd monotonic_dc
python main.py
```

This runs the DC model for the C-MAPSS dataset by default. If you want to change some of the default variables, for example, if you want to enable the Bayesian Optimization algorithm and not rely on the existing hyperparameters, and to train the DC model on the MIMIC-III dataset run the command:

`python main.py --bayesian_opt True --mimic True`

See the `main.py` file for different existing variables and options.

### Results

The results are saved inside the directory `../Monotonic_DC/monotonic_dc/results/`The clustering results and the survivability plots (produced by the Kaplan-Meier method) for 10 trajectories of the C-MAPSS dataset are shown below:

![alt text](https://github.com/panoskom/ISTRUST_model/blob/main/Figs/Clustering_results_Survivability_plot.jpg)

>**Note**
>The results may be slightly different for different hardware setups. Additionally, varying tuned hyperparameters may be used after running the Bayesian Optimization algorithm on different hardware. This explains why we presented in the paper the mean and variance of the losses over 10 independent runs of the code. The corresponding figures come after setting the seeding of the algorithm, which is different depending on the computer system.

