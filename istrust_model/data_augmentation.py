import torch
import torchvision.transforms as T
import random
import data
import params
import utils
import torchvision
from tqdm import tqdm


def augment_data(filename_dataset, N, cpu_storage=True, sigma_rul=0.01):
    """
        Function to create an augmented dataset
        It is saved with name filename_dataset_training_set_augmented, and only augments the training set
        
        Args:
            filename_dataset: string
                filename of the dataset without extension (e.g. window_size_7_test_0)
            N: int
                number of times you want to augment the entire dataset with the same (however randomized) augmentations
            cpu_storage: bool
                if True, stores the dataset on the cpu, else it is stored on cuda if available
            sigma_rul: float
                standard deviation of the noise added to the RUL
    """
    # Initialize devices
    devices = utils.Devices(cpu_storage=cpu_storage)
    devices.print_devices()
    # Load dataset
    data_params = params.DataParams(f'dataset_processed/{filename_dataset}_data_params.json')
    data_params.image_width = data_params.image_width_aug
    data_params.image_height = data_params.image_height_aug
    training_set = data.BuildDataset(*torch.load(f'dataset_processed/{filename_dataset}_training_set.pth',
                                                 map_location=devices.device_storage))

    # Antialias parameter is not available in torchvision version smaller than or equal to 13
    torchvision_version, torchvision_subversion = torchvision.__version__.split('.')[:2]
    torchvision_version, torchvision_subversion = int(torchvision_version), int(torchvision_subversion)
    if torchvision_version == 0 and torchvision_subversion <= 13:
        resized_crop = T.RandomResizedCrop((data_params.image_height, data_params.image_width),
                                           scale=(0.5, 1.0), ratio=(7/8, 8/7))
    else:
        resized_crop = T.RandomResizedCrop((data_params.image_height, data_params.image_width),
                                           scale=(0.5, 1.0), ratio=(7/8, 8/7), antialias=True)

    # Initialize the (type of) augmentations
    augmentations = T.Compose([
        resized_crop,
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation([-5, 5]),
    ])

    # Initialize list
    pictures_train_augmented = []
    rul_target_train_augmented = []
    times_train_augmented = []
    start_indices_train_augmented = []
    end_indices_train_augmented = []
    experiment_train_augmented = []

    # Loop through the dataset for N times to apply augmentations
    current_len = 0
    for _ in tqdm(range(N), leave=True):
        rul_target_train_current = training_set.RUL_target.detach()
        times_train_current = training_set.times.detach()
        pictures_train_current = training_set.pictures.detach()
        start_indices_train_current = training_set.start_indices.detach()
        end_indices_train_current = training_set.end_indices.detach()
        experiment_train_current = training_set.experiment.detach()
        if training_set.sample_from_tensor:
            pictures_train_augmented.append(utils.normalize_images(augmentations(pictures_train_current),
                                                                   data_params.normalize_value_dataset).cpu())
            rul_target_train_augmented.append(rul_target_train_current + random.normalvariate(0, sigma_rul))
            times_train_augmented.append(times_train_current)
            experiment_train_augmented.append(experiment_train_current)
            # add n times the length of pictures_train for start and end indices
            start_indices_train_augmented.append(start_indices_train_current + current_len)
            end_indices_train_augmented.append(end_indices_train_current + current_len)
            current_len += len(pictures_train_current)
        else:
            raise NotImplementedError("data augmentation currently not implemented for sample_from_tensor == False")
        del pictures_train_current
        del rul_target_train_current
        del times_train_current

    # Concatenate all the lists to a tensor
    pictures_train_augmented = torch.cat(pictures_train_augmented)
    rul_target_train_augmented = torch.cat(rul_target_train_augmented).to(torch.float32)
    times_train_augmented = torch.cat(times_train_augmented)
    start_indices_train_augmented = torch.cat(start_indices_train_augmented)
    end_indices_train_augmented = torch.cat(end_indices_train_augmented)
    experiment_train_augmented = torch.cat(experiment_train_augmented)
    # Save the data
    data_params.save(f'dataset_processed/{filename_dataset}_data_params_aug.json')
    training_set_augmented = data.BuildDataset(pictures_train_augmented, rul_target_train_augmented,
                                               times_train_augmented, start_indices_train_augmented,
                                               end_indices_train_augmented, experiment_train_augmented,
                                               training_set.sample_from_tensor)
    del pictures_train_augmented
    training_set_augmented.save(f'dataset_processed/{filename_dataset}_training_set_augmented.pth')
    del training_set_augmented
