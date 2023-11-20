import json


def _check_required_params(params_dict, required_parameters, filename_load):
    for required_param in required_parameters:
        if required_param not in params_dict.keys():
            raise NameError(required_param + f' is undefined\n  Check {filename_load}')


class ModelParams:
    """
    Hyperparameters related to the model

    Attributes:
        filename_load: string
            location and filename from which to load the json file
        d_model: int
            model size
        d_value: int
            value size in multi-head attention, if None d_value=d_model
        n_head_spatial: int
            number of heads in spatial transformer-encoder
        n_head_temporal: int
            number of heads in temporal transformer-encoder
        patch_width: int
            width of a patch in pixels
        patch_height: int
            height of a patch in pixels
        d_predictor: int = 1
            number of values to predict (regression)
    """
    def __init__(self, filename='hyperparams/model_params.json'):
        self.filename_load = filename
        self.d_model = 96
        self.d_value = None
        self.n_head_spatial = 6
        self.n_head_temporal = 6
        self.scaling_factor_weights = 0.67
        self.eps = 1e-09
        self.patch_width = 16
        self.patch_height = 16
        self.d_predictor = 1

        # Load attributes from json file
        with open(self.filename_load) as model_params_json:
            model_params_dict = json.load(model_params_json)
        _check_required_params(model_params_dict, list(self.__dict__.keys())[1:], self.filename_load)
        self.__dict__.update(model_params_dict)

        if self.d_value is None:
            self.d_value = self.d_model
        else:
            assert type(self.d_value) == int
            self.d_value = self.d_value

    def save(self, filename_save=None):
        if filename_save is None:
            filename_save = self.filename_load
        model_params_dict = self.__dict__
        model_params_dict.pop('filename_load', None)
        model_params_json = json.dumps(model_params_dict, indent=4)
        with open(filename_save, "w") as model_params:
            model_params.write(model_params_json)


class TrainingParams:
    """
        Hyperparameters related to the training loop

        Attributes:
            filename_load: string
                location and filename from which to load the json file
            batch_size: int
                batch size of datasets during training inference
            test_batch_size: int
                batch size of datasets during validation/testing inference
            skip_images: int
                s_skip as described in paper
            shuffle_train: bool
                if True, shuffle the training set
            num_epochs: dict
                contains the num epochs for each stage (stage "s1" and "s2")
            lr: dict
                contains the learning rates for each stage (stage "s1" and "s2")
            momentum: float
                momentum of the SGD optimizer
            dropout: float
                contains the dropout probabilities for each stage (stage "s1" and "s2")
            temp_supcr: float
                softmax temperature of the SupCr loss function during first stage training
            thresh_select_best: float
                threshold value of SupCR loss of non-augmented training set during the first stage training
                to select the best epoch to use for the second stage training
        """
    def __init__(self, filename='hyperparams/training_params.json'):
        self.filename_load = filename
        self.batch_size = int()
        self.test_batch_size = int()
        self.skip_images = int()
        self.shuffle_train = bool()
        self.num_epochs = dict()
        self.lr = dict()
        self.momentum = float()
        self.dropout = dict()
        self.temp_supcr = float()
        self.thresh_select_best = float()

        # Load attributes from json file
        with open(self.filename_load) as training_params_json:
            training_params_dict = json.load(training_params_json)
        _check_required_params(training_params_dict, list(self.__dict__.keys())[1:], self.filename_load)
        self.__dict__.update(training_params_dict)

    def save(self, filename_save=None):
        if filename_save is None:
            filename_save = self.filename_load
        training_params_dict = self.__dict__
        training_params_dict.pop('filename_load', None)
        training_params_json = json.dumps(training_params_dict, indent=4)
        with open(filename_save, "w") as training_params:
            training_params.write(training_params_json)


class DataParams:
    """
        Hyperparameters related to the dataset

        Attributes:
            filename_load: string
                location and filename from which to load the json file
            image_width: int
            image_height: int
            n_features: int
            min_train: float
                minimum pixel value of the training set
            max_train: float
                maximum pixel value of the training set
            testing_set_exists: bool
                whether a testing set exists or there is only a validation set
            normalize_value_dataset: int
                the maximum possible pixel value according to the integer precision (typically 255)
            mean: float
                mean of the training set or the augmented training set if it exists
            std: float
                standard deviation of the training set or the augmented training set if it exists
            sample_from_tensor: bool
                if True, samples from a common tensor for each window
                for in depth behaviour see data.BuildDataset
            n_samples_per_experiment: int
                currently the number of camera views
    """
    def __init__(self, filename='hyperparams/data_params_init.json'):
        self.filename_load = filename
        self.window_size = int()
        self.image_width = 320
        self.image_height = 640
        self.image_width_aug = 240
        self.image_height_aug = 480
        self.n_features = 1
        self.min_train = 0
        self.max_train = 255
        self.testing_set_exists = True
        self.validation_set_exists = True
        self.normalize_value_dataset = 255
        self.mean = 0.0
        self.std = 1.0
        self.sample_from_tensor = True
        self.n_samples_per_experiment = 2

        # Load attributes from json file
        with open(self.filename_load) as data_params_json:
            data_params_dict = json.load(data_params_json)
        _check_required_params(data_params_dict, list(self.__dict__.keys())[1:], self.filename_load)
        self.__dict__.update(data_params_dict)

        self.n_patches_width = None
        self.n_patches_height = None
        self.n_patches = None

    def from_model(self, model_params):
        self.n_patches_width = self.image_width // model_params.patch_width
        self.n_patches_height = self.image_height // model_params.patch_height
        self.n_patches = self.n_patches_width * self.n_patches_height

    def save(self, filename_save=None):
        if filename_save is None:
            filename_save = self.filename_load
        data_params_dict = self.__dict__
        data_params_dict.pop('filename_load', None)
        data_params_dict.pop('n_patches_width', None)
        data_params_dict.pop('n_patches_height', None)
        data_params_dict.pop('n_patches', None)
        data_params_json = json.dumps(data_params_dict, indent=4)
        with open(filename_save, "w") as data_params:
            data_params.write(data_params_json)
