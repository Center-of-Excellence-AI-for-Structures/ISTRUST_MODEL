# Suppress numba warning message (requires that numba version is <= 0.58)
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import argparse
import torch
from run_istrust import RunISTRUST
from data_gen import create_dataset
from data_augmentation import augment_data
from utils import search_folder, str2bool, test_samples2list, check_args


def run_process(args):
    """
        Run the entire process

        Args:
            args: arguments originating from parser.parse_args()
    """

    # Check whether boolean combinations are allowed/make sense
    check_args(args)

    # Get arguments
    has_validation = args.has_validation
    create_data = args.create_data
    create_augmented_data = args.create_augmented_data
    n_aug = args.n_aug
    data_augmentation = args.data_augmentation
    train = args.train
    test = args.test
    train_stage_2 = args.train_stage_2
    export_attention_weights = args.export_attention_weights
    export_umap = args.export_umap
    test_samples = args.test_samples

    for test_sample in test_samples:
        # Get (filename of) dataset
        if create_data:
            print("Creating dataset")
            filename_dataset = create_dataset(test_sample, has_validation=has_validation)
        else:
            filename_dataset = f"window_size_7_test_{test_sample}"

        if create_augmented_data:
            print("\nAugmenting dataset")
            augment_data(filename_dataset, n_aug)

        print("\nInitializing ISTRUST")
        istrust = RunISTRUST(
            name_data=filename_dataset,
            cpu_storage=False,
            apply_augmentation=data_augmentation
        )

        if train:
            istrust.train_two_stage()

        if train_stage_2:
            folder_name_test = search_folder(filename_dataset)

            # Actual training
            istrust.train_stage_2(folder_name_test)

        if test:
            folder_name_test = search_folder(filename_dataset)

            # Actual testing
            istrust.test(folder_name_test, export_attention_weights=export_attention_weights, export_umap=export_umap)


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print("Code running...\n")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_samples",
        default=list(range(1, 7)),
        type=test_samples2list,
        help="Test sample to use if not using entire range: choose from 1-2-3-4-5-6, by default all test samples "
             "are included (range from 1 to 6), default=list(range(1, 7))",
    )
    parser.add_argument(
        "--has_validation",
        default=False,
        type=str2bool,
        help="if True, keeps the validation set separately (if False, used for training), default=False",
    )
    parser.add_argument(
        "--create_data",
        default=True,
        type=str2bool,
        help="if True, the raw data is processed to a pytorch dataset, "
             "if False, it is assumed that this has already been done by the user. "
             "The dataset is stored in the dataset_processed folder, default=True"
    )
    parser.add_argument(
        "--create_augmented_data",
        default=True,
        type=str2bool,
        help="Whether we need to create augmented data. After creating them, set this to False to save time, "
             "default=True"
    )
    parser.add_argument(
        "--n_aug",
        default=65,
        type=int,
        help="factor by which you want to increase dataset size by the means of data augmentation ('N' in paper), "
             "note that this is stochastic and will change every time it is ran again, default=65",
    )
    parser.add_argument(
        "--data_augmentation",
        default=True,
        type=str2bool,
        help="if True, uses (NOT creates) the augmented training dataset during first stage training, default=True",
    )
    parser.add_argument(
        "--train",
        default=False,
        type=str2bool,
        help="if True, trains the model (useful to set to False if there is already a trained model), default=False",
    )
    parser.add_argument(
        "--test",
        default=True,
        type=str2bool,
        help="whether or not to test the model, default=True",
    )
    parser.add_argument(
        "--train_stage_2",
        default=False,
        type=str2bool,
        help="if True, redoes only the second stage training based on existing first stage training results, "
             "default=False",
    )
    parser.add_argument(
        "--export_attention_weights",
        default=False,
        type=str2bool,
        help="if True, exports all attention weights, default=False",
    )
    parser.add_argument(
        "--export_umap",
        default=False,
        type=str2bool,
        help="if True, exports the UMAP of the training sets, is independent of whether or not train is True, "
             "caution, this is computationally expensive and might take up to a few minutes extra, default=False",
    )

    run_process(parser.parse_args())
