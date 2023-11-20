import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import argparse


def str2bool(v):
    """
        Converts argument to boolean value
        Taken from: https://stackabuse.com/bytes/parsing-boolean-values-with-argparse-in-python/

        Args:
            v: str
                string argument specified by user in terminal

        Returns: bool
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def test_samples2list(test_samples):
    """
        Converts argument of type str or int to iterable (list)
    """
    if isinstance(test_samples, str):
        test_samples = list([int(test_samples)])
    elif test_samples is None:
        test_samples = [None]
    return test_samples


def normalize_images(images, normalize_value_dataset):
    """
        Normalizes images between zero and a predefined maximum value

        Parameters:
            images: torch.Tensor
                input images to be normalized
            normalize_value_dataset:
                maximum value to which the images shall be normalized

        Returns:
            images: torch.Tensor
                normalized images
    """
    images = images - torch.amin(images, dim=(-2, -1)).unsqueeze(dim=-1).unsqueeze(dim=-1)
    images = torch.div(images, torch.amax(images, dim=(-2, -1)).unsqueeze(dim=-1).unsqueeze(dim=-1))
    images = (normalize_value_dataset * images).to(torch.uint8)
    return images


def mean_std_calculator(images, normalize_value_dataset):
    """
        Calculates the mean and standard deviation for a large tensor

        Parameters:
            images: torch.Tensor
                input images for which the mean and std should be calculated
            normalize_value_dataset: int

    """
    images = images.to(torch.float32)/normalize_value_dataset
    mean, std = 0., 0.

    for i in range(len(images)):
        mean += torch.mean(images[i])
        std += torch.mean(images[i]**2)

    mean = mean / len(images)
    std = (std/len(images) - mean**2)**0.5
    return mean, std


class Devices:
    """
        Devices class containing the model device and storage device

        Parameters:
            cpu_storage: bool
                if True: use cpu to store dataset, regardless of cuda availability
                if False: use cuda to store dataset, if cuda is available
    """
    def __init__(self, cpu_storage: bool = False):
        self.device_model = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if cpu_storage:
            self.device_storage = 'cpu'
        else:
            self.device_storage = self.device_model

    def print_devices(self):
        print(f'Model device is {self.device_model}\nStorage device is {self.device_storage}')


def plot_to_tensorboard(writer, fig, filename, global_step=0):
    """
        Takes a matplotlib figure handle and converts it using
        canvas and string-casts to a numpy array that can be
        visualized in TensorBoard using the add_image function

        Edited from: https://martin-mundt.com/tensorboard-figures/

        Parameters:
            writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
            fig (matplotlib.pyplot.fig): Matplotlib figure handle.
            filename: string
            global_step: int
    """

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    img = np.swapaxes(img, 0, 2)  # if your TensorFlow + TensorBoard version are >= 1.8
    img = np.swapaxes(img, 1, 2)  # if your TensorFlow + TensorBoard version are >= 1.8

    # Add figure in numpy "image" to TensorBoard writer
    writer.add_image(filename, img, global_step=global_step)
    plt.close(fig)


def confidence_intervals(out, confidence=0.95):
    """
        Calculate the confidence interval the MC RULs at a given timestep

        Parameters:
            out: numpy 1D array
                the predicted RULs for a certain timestep for all MC samples
            confidence: float
                confidence level of the confidence interval

        Returns:
            CI.confidence_interval: ConfidenceInterval
                The bootstrap confidence interval as an instance of collections.namedtuple with attributes low and high.
            m: float
                mean of RULs for that specific timestep, can thus be interpreted as the predicted RUL
    """
    m = out.mean()
    out = (out,)
    CI = stats.bootstrap(out, np.std, confidence_level=confidence, method='BCa')
    return CI.confidence_interval, m


def search_folder(filename_dataset):
    """
        Searches for the folder in which the data resulting from the training of a certain test sample is stored

        Args:
            filename_dataset: str
                filename of the dataset (e.g. window_size_7)
        Returns:
            folder_name_test: str
                the directory in which the relevant data is stored (e.g. runs/window_size_7_test_6-20_09_23-12_26)
    """
    name_folder = None
    for name_folder in os.listdir(f"runs"):
        name_folder = str(name_folder)
        if filename_dataset in name_folder:
            break
    if name_folder is not None:
        if filename_dataset in name_folder:
            folder_name_test = name_folder
        else:
            raise ValueError(f"Folder for {filename_dataset} not found")
    else:
        raise ValueError("./runs directory is empty")

    folder_name_test = "runs/" + folder_name_test
    return folder_name_test
