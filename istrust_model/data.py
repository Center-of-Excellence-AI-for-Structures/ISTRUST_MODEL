from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from params import DataParams


class BuildDataset(Dataset):
    """
        Main class to handle the dataset, builds on top of torch.utils.data.Dataset

        Args:
            pictures: tensor
                tensor in which the input images are
                if sample_from_tensor is True:
                    input images are not yet sampled in windows, but it is just a tensor containing each image once
                else:
                    input images are already sampled in windows
            rul_target: tensor
                tensor containing all target RULs for each window
            times: tensor
                tensor contain all timestamps for each window
            start_indices: tensor
                tensor containing start indices for each window, especially relevant if sample_from_tensor is True (but
                still rqrd if False)
            end_indices: tensor
                end incices for each window, similar to start indices
            experiment: tensor
                experiment indices for each window
            sample_from_tensor: bool
                for behaviour see above
            skip_images: int
                skip images as described in Figure 6 in paper

    """
    def __init__(self, pictures, rul_target, times, start_indices, end_indices, experiment, sample_from_tensor=True,
                 skip_images=None):
        self.RUL_target = rul_target
        self.pictures = pictures
        self.times = times
        self.start_indices = start_indices
        self.end_indices = end_indices
        self.experiment = experiment
        self.sample_from_tensor = sample_from_tensor
        if skip_images is not None:
            assert type(skip_images) == int
            self.skip_images = skip_images
        else:
            self.skip_images = skip_images

    def __len__(self):
        return len(self.RUL_target)

    def __getitem__(self, idx):
        if self.sample_from_tensor:
            if self.skip_images is None:
                return [self.pictures[int(self.start_indices[idx]):int(self.end_indices[idx])],
                        self.RUL_target[idx], self.times[idx], self.experiment[idx]]
            else:
                return [self.pictures[int(self.start_indices[idx]):int(self.end_indices[idx]):self.skip_images],
                        self.RUL_target[idx], self.times[idx], self.experiment[idx]]
        else:
            if self.skip_images is None:
                return [self.pictures[idx],
                        self.RUL_target[idx], self.times[idx], self.experiment[idx]]
            else:
                return [self.pictures[idx::self.skip_images],
                        self.RUL_target[idx], self.times[idx], self.experiment[idx]]

    def to(self, to_key):
        """
            Makes it possible to use default pytorch .to(...) function on the entire dataset
        """
        self.pictures = self.pictures.to(to_key)
        self.RUL_target = self.RUL_target.to(to_key)
        self.times = self.times.to(to_key)
        self.start_indices = self.start_indices.to(to_key)
        self.end_indices = self.end_indices.to(to_key)
        self.experiment = self.experiment.to(to_key)
        return self

    def set_skip_images(self, skip_images):
        """
            Sets the number of images to skip when sampling windows (for more info see Figure 6 in paper)
        """
        self.skip_images = skip_images

    def save(self, file_location):
        """
            Saves the entire dataset as a pth tensor object file
        """
        torch.save([self.pictures, self.RUL_target, self.times, self.start_indices, self.end_indices, self.experiment,
                    self.sample_from_tensor], file_location)


def load_dataset(filename_dataset, batch_size, batch_size_val, use_augmentation, devices, shuffle_train,
                 skip_images=None):
    """
        Function that loads all datasets as a BuildDataset object (see above) instead of only a single one

        Args:
            filename_dataset: string
                filename of the dataset, without extensions (e.g. window_size_7)
            batch_size: int
                batch size for training_set
            batch_size_val: int
                batch size for training_set_eval, validation_set, testing_set
            use_augmentation: bool
                if True, uses the augmented training set, does not apply any data augmentation
            devices: utils.Devices object
                devices used to store dataset
            shuffle_train: bool
                if True, the training set is shuffled at every epoch
            skip_images: int
                skip images as described in Figure 6 in paper
    """
    if use_augmentation:
        # Load data params of augmented training set ('augmented' only influences the image width and height)
        data_params = DataParams(f'dataset_processed/{filename_dataset}_data_params_aug.json')
        # Load augmented training set (used for training)
        training_set = BuildDataset(*torch.load(f'dataset_processed/{filename_dataset}_training_set_augmented.pth',
                                                map_location=devices.device_storage), skip_images)
        training_set = DataLoader(training_set, batch_size=batch_size, shuffle=shuffle_train)
        # Load non-augmented training set (used for evaluation)
        training_set_eval = BuildDataset(*torch.load(f'dataset_processed/{filename_dataset}_training_set.pth',
                                                     map_location=devices.device_storage), skip_images)
        training_set_eval = DataLoader(training_set_eval, batch_size=batch_size_val, shuffle=False)
    else:
        # Load data params of training set ('augmented' only influences the image width and height)
        data_params = DataParams(f'dataset_processed/{filename_dataset}_data_params.json')
        # Load non-augmented training set as training set as well (used for training)
        training_set = BuildDataset(*torch.load(f'dataset_processed/{filename_dataset}_training_set.pth',
                                                map_location=devices.device_storage), skip_images)
        training_set = DataLoader(training_set, batch_size=batch_size, shuffle=shuffle_train)
        # Load non-augmented training set (used for evaluation)
        training_set_eval = BuildDataset(*torch.load(f'dataset_processed/{filename_dataset}_training_set.pth',
                                                     map_location=devices.device_storage), skip_images)
        training_set_eval = DataLoader(training_set_eval, batch_size=batch_size_val, shuffle=False)

    # Load validation set is it exists
    if data_params.validation_set_exists:
        validation_set = BuildDataset(*torch.load(f'dataset_processed/{filename_dataset}_validation_set.pth',
                                                  map_location=devices.device_storage), skip_images)
        validation_set = DataLoader(validation_set, batch_size=batch_size_val, shuffle=False)
    else:
        validation_set = None

    # Load testing set if it exists
    if data_params.testing_set_exists:
        testing_set = BuildDataset(*torch.load(f'dataset_processed/{filename_dataset}_testing_set.pth',
                                               map_location=devices.device_storage), skip_images)
        testing_set = DataLoader(testing_set, batch_size=batch_size_val, shuffle=False)
    else:
        testing_set = None
    return training_set, training_set_eval, validation_set, testing_set, data_params
