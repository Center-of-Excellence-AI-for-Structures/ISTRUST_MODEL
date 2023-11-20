import torch
import params
import utils
import os
from PIL import Image
import numpy as np
import torchvision
import data


def load_raw_dataset(directory_dataset="dataset", filename_data_params="hyperparams/data_params_init.json"):
    """
        Loads the raw dataset for preprocessing

        Args:
            directory_dataset: string
            filename_data_params:string

        Returns:
            dataset_same_time_split: list
                each index contains a list of RULs, raw input images and experiment numbers
                each index in these lists is of format [rul: float, image: tensor(uint8), experiment_number:int]
            data_params: params.DataParams
    """
    # Load data params
    data_params = params.DataParams(filename_data_params)

    current_file = ""
    dataset = []
    ruls = []
    resize = torchvision.transforms.Resize((data_params.image_height, data_params.image_width), antialias=True)
    counter = -1
    experiments = []

    # Loop through the entire directory
    for i in os.listdir(directory_dataset):
        filename = directory_dataset + "/" + i
        i = i[:-5].split("_")
        rul = float(i[-2]) + float("0."+i[-1])
        ruls.append(rul)
        # Exclude RUL < 4000 (otherwise large bias) and A025 because it had inconsistent step sizes
        if rul < 4000 and (i[0] + i[1])[:-1] != "A025":
            # Check if you're still handling the same experiment and view
            if i[0] + i[1] != current_file:
                current_file = i[0] + i[1]
                counter += 1
                # Store the data to dataset list
                experiments.append([counter])
                dataset.append([[rul, torch.squeeze(resize(torch.unsqueeze(
                    torch.Tensor(np.array(Image.open(filename))), 0))).to(torch.uint8)]])
            else:
                # Store the data to dataset list
                experiments[-1].append(counter)
                dataset[-1].append([rul, torch.squeeze(resize(torch.unsqueeze(
                    torch.Tensor(np.array(Image.open(filename))), 0))).to(torch.uint8)])

    dataset_same_time_split = []
    for i in range(len(dataset)):
        dataset[i] = list(sorted(dict(dataset[i]).items()), )[::-1]
        # Exclude the first 8/9 (even number rqrd) images because then the experiment still has to stabilise
        if len(dataset[i]) % 2 == 0:
            dataset[i] = dataset[i][8:]
            experiments[i] = experiments[i][8:]
        else:
            dataset[i] = dataset[i][9:]
            experiments[i] = experiments[i][9:]
        # Put experiment number in dataset list
        for j in range(len(dataset[i])):
            dataset[i][j] = [dataset[i][j][0], dataset[i][j][1], experiments[i][j]]
        # Remove duplicate time values (two pictures are taken at the same time)
        dataset_same_time_split.append(dataset[i][1::2])
    return dataset_same_time_split, data_params


def create_targets(dataset, sample_from_tensor, data_params):
    """
        Creates the targets used to build the dataset as separate tensors

        Args:
            dataset: list
                training/validation/testing/... dataset obtained with function create_dataset
            sample_from_tensor: bool
                if True, samples from a common tensor for each window
                for in depth behaviour see data.BuildDataset
            data_params: params.DataParams

        Returns:
            pictures_list: tensor
            rul_target: tensor
            times_list: tensor
            start_indices: tensor
            end_indices: tensor
            experiments_list: tensor
    """
    # Create all the lists initialised below
    counter = 0
    pictures_list = []
    times_list = []
    rul_list = []
    rul_target = []
    start_indices = []
    experiments_list = []
    for i in range(len(dataset)):
        if sample_from_tensor:
            for j in range(len(dataset[i])):
                dataset[i][j] = list(dataset[i][j])
                rul_list.append(dataset[i][j][0])
                times_list.append(round(dataset[i][0][0] - dataset[i][j][0], 2))
                pictures_list.append(torch.unsqueeze(dataset[i][j][1].detach(), 0))
                experiments_list.append(dataset[i][j][2])
        else:
            for j in range(len(dataset[i]) - data_params.window_size):
                current_pictures = []
                dataset[i][j] = list(dataset[i][j])
                rul_target.append(dataset[i][j + data_params.window_size][0])
                times_list.append(dataset[i][0][0] - rul_target[-1])
                for m in range(data_params.window_size):
                    current_pictures.append(torch.unsqueeze(torch.unsqueeze(dataset[i][j + m][1].detach(), 0), 0))
                pictures_list.append(torch.unsqueeze(torch.cat(current_pictures), 0))
                experiments_list.append(dataset[i][j][2])
        for j in range(len(dataset[i]) - data_params.window_size):
            start_indices.append(counter + j)
        counter += len(dataset[i])
    del dataset
    start_indices = torch.Tensor(start_indices).to(torch.int32)
    end_indices = (start_indices+data_params.window_size).to(torch.int32)

    # Create the targets (they are tensors)
    if sample_from_tensor:
        rul_target = []
        times_target = []
        experiments_target = []
        for i in range(len(start_indices)):
            current_rul = rul_list[end_indices[i]:end_indices[i]+1]
            rul_target.append(torch.unsqueeze(torch.Tensor(current_rul), dim=0))
            times_target.append(times_list[end_indices[i]])
            experiments_target.append(experiments_list[end_indices[i]])
        times_list = torch.Tensor(times_target)
        experiments_list = torch.Tensor(experiments_target)
    else:
        times_list = torch.Tensor(times_list)
        experiments_list = torch.Tensor(experiments_list)
    rul_target = torch.cat(rul_target)
    pictures_list = torch.cat(pictures_list)
    pictures_list = pictures_list.unsqueeze(dim=-3)
    return pictures_list, rul_target, times_list, start_indices, end_indices, experiments_list


def create_dataset(test_sample=None, exclude_sample_7=True, has_validation=True):
    """
        Function that actually creates and saves all the training, validation and testing sets
        Validation experiment is A003, all others are used for testing/training

        Args:
            test_sample: int
                experiment number of the test sample you want to use (or None if you want to use no test sample)
            exclude_sample_7: int
                If True, excludes sample 7, which is excluded due to displacements over time in images
                (sample moves w.r.t. input image)
            has_validation: bool
                If True, to use sample 0 for validation or for training
        Returns:
            filename: str
                the filename of the dataset, e.g. window_size_7_test_0
    """
    dataset_raw, data_params = load_raw_dataset()
    if exclude_sample_7:
        dataset_raw = dataset_raw[:-2]
    # Check if user wants to create testing set and/or validation set as well
    if test_sample is None:
        data_params.testing_set_exists = False
    if not has_validation:
        data_params.validation_set_exists = False

    # Index datasets from the raw dataset
    if data_params.validation_set_exists:
        validation_set = dataset_raw[:data_params.n_samples_per_experiment].copy()
        training_set = dataset_raw[data_params.n_samples_per_experiment:].copy()
    else:
        training_set = dataset_raw.copy()
    if data_params.testing_set_exists:
        testing_set = training_set[test_sample * data_params.n_samples_per_experiment:
                                   (test_sample + 1) * data_params.n_samples_per_experiment].copy()
        del training_set[test_sample * data_params.n_samples_per_experiment:
                         (test_sample + 1) * data_params.n_samples_per_experiment]

    # Put the datasets in the correct class format we want (which is data.BuildDataset)
    training_set = data.BuildDataset(*create_targets(training_set, data_params.sample_from_tensor, data_params),
                                     data_params.sample_from_tensor)
    if data_params.validation_set_exists:
        validation_set = data.BuildDataset(*create_targets(validation_set, data_params.sample_from_tensor, data_params),
                                           data_params.sample_from_tensor)
    if data_params.testing_set_exists:
        testing_set = data.BuildDataset(*create_targets(testing_set, data_params.sample_from_tensor, data_params),
                                        data_params.sample_from_tensor)

    # Apply histogram equalization and normalize images between 0 and normalize value dataset
    training_set.pictures = utils.normalize_images(
        torchvision.transforms.functional.equalize(training_set.pictures), data_params.normalize_value_dataset)
    if data_params.validation_set_exists:
        validation_set.pictures = utils.normalize_images(
            torchvision.transforms.functional.equalize(validation_set.pictures), data_params.normalize_value_dataset)
    if data_params.testing_set_exists:
        testing_set.pictures = utils.normalize_images(
            torchvision.transforms.functional.equalize(testing_set.pictures), data_params.normalize_value_dataset)

    # Normalize RUL
    min_train = 0.
    max_train = float(torch.max(training_set.RUL_target[:, 0]))
    training_set.RUL_target = training_set.RUL_target / max_train
    if data_params.validation_set_exists:
        validation_set.RUL_target = (validation_set.RUL_target - min_train) / max_train
    if data_params.testing_set_exists:
        testing_set.RUL_target = (testing_set.RUL_target - min_train) / max_train

    # Calculate mean and std
    mean, std = utils.mean_std_calculator(images=training_set.pictures,
                                          normalize_value_dataset=data_params.normalize_value_dataset)

    # Check if save folder "dataset_processed" exist, create it if it doesn't
    if not os.path.exists("dataset_processed"):
        os.mkdir("dataset_processed")

    # Save all data
    filename = f"window_size_{data_params.window_size}"
    if data_params.validation_set_exists:
        test_sample += 1
    if data_params.testing_set_exists:
        filename = filename + f"_test_{test_sample}"
    training_set.save(f"dataset_processed/{filename}_training_set.pth")
    if data_params.validation_set_exists:
        validation_set.save(f"dataset_processed/{filename}_validation_set.pth")
    if data_params.testing_set_exists:
        testing_set.save(f"dataset_processed/{filename}_testing_set.pth")
    data_params.min_train = min_train
    data_params.max_train = max_train
    data_params.mean = float(mean)
    data_params.std = float(std)
    data_params.save(f"dataset_processed/{filename}_data_params.json")
    print(f"Dataset saved with filename {filename}")
    return filename
