import contrastive_regression
from model import Encoder, Predictor
import torch
import utils
import torch.nn as nn
import keyboard
from torch.utils.tensorboard import SummaryWriter
import data
import params
import attention
import matplotlib.pyplot as plt
from umap_reduction import UmapReduction
import numpy as np
import pandas as pd
import os
import time
import warnings
from tqdm import tqdm


def weight_loss(test_loss, last_batch_size, first_batch_size):
    """
        Weights the loss of the last batch according to its relative size

        Args:
            test_loss: numpy 1D array
                array of losses
            last_batch_size: int
                batch size of last batch
            first_batch_size: int
                batch size of other batches

        Returns:
            test_loss: float
                weighted test loss
    """
    weights_loss = np.ones_like(test_loss)
    weights_loss[-1] = last_batch_size / first_batch_size
    test_loss = np.average(test_loss, weights=weights_loss)
    return test_loss


class RunISTRUST(nn.Module):
    """
        Acts as the main class to perform operations with the model, like training and testing

        Parameters:
            name_data: str
                name of dataset (e.g. window_size_7), looks in dataset_processed folder by default
            cpu_storage: bool
                if True, it will store the dataset on the cpu, otherwise on cuda if available
            apply_augmentation: bool
                if True, it will use the augmented training set for stage 1 training
            filename_model_params (optional): string
                a custom filename (location from project root) to load the model params json from
            filename_training_params (optional): string
                a custom filename (location from project root) to load the training params json from
    """

    def __init__(self, name_data, cpu_storage: bool = False, apply_augmentation: bool = True,
                 filename_model_params=None, filename_training_params=None):
        super().__init__()
        # Initialize devices
        self.devices = utils.Devices(cpu_storage=cpu_storage)
        self.devices.print_devices()

        self.interrupt_code_key = "ctrl+shift+f12"

        # Load model related hyperparams
        if filename_model_params is None:
            self.model_params = params.ModelParams()
        else:
            self.model_params = params.ModelParams(filename_model_params)

        # Load training related hyperparams
        if filename_training_params is None:
            self.training_params = params.TrainingParams()
        else:
            self.training_params = params.TrainingParams(filename_training_params)

        # Data related attributes initialization
        self.folder_name_data = name_data
        self.apply_augmentation = apply_augmentation
        self.training_set = None
        self.training_set_eval = None
        self.validation_set = None
        self.testing_set = None
        self.data_params = None

        # Loss functions
        self.loss = dict()
        self.loss["s1"] = contrastive_regression.SupCR(temp=self.training_params.temp_supcr)
        self.loss["s2"] = nn.L1Loss()
        self.loss_smoothed = []  # Smooth loss of training eval dataset used to determine early stopping

        # Initialize encoder and predictor and dropout in between them
        self.encoder = None
        self.predictor = None

        # Parameters to be used during training
        self.logdir = dict()
        self.writer = None
        self.losses = dict()
        self.losses["s1"] = pd.DataFrame(index=range(self.training_params.num_epochs["s1"]),
                                         columns=["training", "training eval",
                                                  "validation", "testing"])
        self.losses["s2"] = pd.DataFrame(index=range(self.training_params.num_epochs["s2"]),
                                         columns=["training", "training eval",
                                                  "validation", "testing"])
        self.results = dict()

    def _check_stage(self, stage):
        """
            Checks whether the stage is currently supported, raises error if not

            Args:
                stage: str
                    training stage

            Returns:
                None
        """
        if stage not in ["s1", "s2"]:
            raise NotImplementedError(f"Stage {stage} is currently not supported, "
                                      f"only stage 's1' and 's2' are supported")

    def _init_encoder(self, stage, model_path=None):
        """
            Initializes the encoder module

            Args:
                stage: str
                    the current stage of the training process, only influences the dropout probability
                model_path: str
                    if specified, determines the model path (from project root) from which the weights are loaded
                    else, the weights will be initialized following the weight initialization scheme in model.py

            Returns:
                encoder: encoder class object from model.py
                    encoder used to obtain the spatial encoded embeddings
        """
        self._check_stage(stage)
        encoder = Encoder(data_params=self.data_params,
                          model_params=self.model_params,
                          training_params=self.training_params,
                          stage=stage)
        encoder = encoder.to(self.devices.device_model)
        if model_path is not None:
            encoder.load_state_dict(torch.load(model_path))
        encoder.train()
        return encoder

    def _init_predictor(self, model_path=None):
        """
            Initializes the encoder module

            Args:
                model_path: str
                    if specified, determines the model path from which the weights are loaded
                    else, the weights will be initialized following the weight initialization scheme

            Returns:
                predictor: predictor class object from model.py
                    predictor used to obtain the RUL from the spatial encoded embeddings
        """
        predictor = Predictor(
            model_params=self.model_params
        )
        predictor = predictor.to(self.devices.device_model)
        if model_path is not None:
            predictor.load_state_dict(torch.load(model_path))
        predictor.train()
        return predictor

    def _init_optimizer(self, stage):
        """
            Initializes the optimizer to be used at the start of a training stage

            Args:
                stage: str
                    the training stage for which the optimizer can be used
        """
        self._check_stage(stage)
        if stage == "s1":
            return torch.optim.SGD(params=self.encoder.parameters(),
                                   lr=self.training_params.lr[stage],
                                   momentum=self.training_params.momentum)
        else:
            return torch.optim.SGD(params=self.predictor.parameters(),
                                   lr=self.training_params.lr[stage],
                                   momentum=self.training_params.momentum)

    def _init_folder(self, folder_name=None):
        """
            Initializes all folders to be used during training
            Saves the folder locations as class attributes

            Args:
                folder_name: str
                    if not None, the logdir will be set to this folder
        """
        # Folder initialization
        if folder_name is None:
            time_str = time.strftime("%d_%m_%y-%H_%M")
            self.logdir["0"] = "runs/" + self.folder_name_data + "-" + time_str
        else:
            self.logdir["0"] = folder_name
        self.logdir["s1"] = self.logdir["0"] + "/stage_1"
        self.logdir["s2"] = self.logdir["0"] + "/stage_2"
        self.logdir["rul_figures"] = self.logdir["0"] + "/rul_figures"
        self.logdir["losses"] = self.logdir["0"] + "/losses"
        self.logdir["attn"] = self.logdir["0"] + "/attn"
        self.logdir["umap"] = self.logdir["0"] + "/umap"

        # Create folders that do not yet exist
        if not os.path.exists("runs"):
            os.mkdir("runs")
        if not os.path.exists(self.logdir["0"]):
            os.mkdir(self.logdir["0"])
            os.mkdir(self.logdir["s1"])
            os.mkdir(self.logdir["s2"])
        if not os.path.exists(self.logdir["rul_figures"]):
            os.mkdir(self.logdir["rul_figures"])
        if not os.path.exists(self.logdir["losses"]):
            os.mkdir(self.logdir["losses"])
        if not os.path.exists(self.logdir["attn"]):
            os.mkdir(self.logdir["attn"])
        if not os.path.exists(self.logdir["umap"]):
            os.mkdir(self.logdir["umap"])

        print(f"Results directory is set to {self.logdir['0']}")

        # Tensorboard initialization
        self.writer = SummaryWriter(self.logdir["0"])
        print(f"For tensorboard run 'tensorboard --logdir={self.logdir['0']} "
              f"--samples_per_plugin images=50' in the terminal")

    def _select_best(self, stage):
        """
            Selects best training epoch, where the best epoch is defined as the epoch with the lowest training eval loss

            Args:
                stage: str

            Returns:
                best_epoch: int
                    epoch with the lowest training eval loss, count starts from 0
        """
        self._check_stage(stage)
        losses = torch.Tensor(self.losses[stage]["training eval"])
        best_epoch = torch.where(losses == torch.min(losses))[0][0]
        print(f"Best epoch found: epoch {best_epoch + 1}")
        return best_epoch

    def _export_losses(self):
        """
            Export all losses to plots in losses folder
        """
        # Load losses
        self.losses["s1"] = pd.read_csv(self.logdir["0"] + "/losses_s1.csv")
        self.losses["s2"] = pd.read_csv(self.logdir["0"] + "/losses_s2.csv")

        # select best epoch for stage 1
        best_epoch = self._select_best("s1")

        # Plot contrastive learning losses (stage 1)
        plt.figure(dpi=50)
        plt.xlabel('epoch')
        plt.ylabel(r'$\mathcal{L}_{SupCR}$')
        plt.plot(range(1, 1 + len(self.losses["s1"]["training"])), self.losses["s1"]["training"],
                 label='Augmented training data')
        plt.plot(range(1, 1 + len(self.losses["s1"]["training eval"])), self.losses["s1"]["training eval"],
                 label='Raw training data')
        if self.data_params.testing_set_exists:
            plt.plot(range(1, 1 + len(self.losses["s1"]["testing"])), self.losses["s1"]["testing"],
                     label='Testing')
        if self.data_params.validation_set_exists:
            plt.plot(range(1, 1 + len(self.losses["s1"]["validation"])), self.losses["s1"]["validation"],
                     label='Validation')
        plt.xticks(range(2, 1 + len(self.losses["s1"]), 2))
        plt.scatter(int(best_epoch + 1), self.losses["s1"]["testing"][int(best_epoch)], label="Best epoch")
        # Place the legend outside the axes to avoid intersection
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
        plt.draw()
        plt.savefig(f"{self.logdir['losses']}/loss_curve_contrastive.pdf")

        best_epoch = self._select_best("s2")

        # Plot the MAE losses (stage 2)
        plt.figure(dpi=50)
        plt.xlabel('epoch')
        plt.ylabel(r'$\mathcal{L}_{MAE}$')
        plt.plot(range(1, 1 + len(self.losses["s2"]["training"])), self.losses["s2"]["training"],
                 label='Augmented training data')
        plt.plot(range(1, 1 + len(self.losses["s2"]["training eval"])), self.losses["s2"]["training eval"],
                 label='Raw training data')
        if self.data_params.testing_set_exists:
            plt.plot(range(1, 1 + len(self.losses["s2"]["testing"])), self.losses["s2"]["testing"],
                     label='Testing')
        if self.data_params.validation_set_exists:
            plt.plot(range(1, 1 + len(self.losses["s2"]["validation"])), self.losses["s2"]["validation"],
                     label='Validation')
        plt.xticks(range(2, 1 + len(self.losses["s2"]), 2))
        plt.scatter(int(best_epoch + 1), self.losses["s2"]["testing"][int(best_epoch)], label="Best epoch")
        # Place the legend outside the axes to avoid intersection
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
        plt.draw()
        plt.savefig(f"{self.logdir['losses']}/loss_curve_MAE.pdf")

    def save_params(self):
        self.model_params.save(self.logdir["0"] + "/model_params.json")
        self.training_params.save(self.logdir["0"] + "/training_params.json")
        if self.data_params is not None:
            self.data_params.save(self.logdir["0"] + "/data_params.json")
        else:
            warnings.warn(f"self.data_params is not defined, skipping save")

    def load_dataset(self, folder_name_data, apply_augmentation=False):
        """
            Loads all the datasets from dataset_processed folder

            Args:
                folder_name_data: str
                    the name of the dataset (e.g. window_size_7_test_2)
                apply_augmentation: bool
                    if True, it loads augmented training set as primary training set
        """
        # Load dataset
        self.training_set, self.training_set_eval, self.validation_set, self.testing_set, self.data_params = \
            data.load_dataset(filename_dataset=folder_name_data,
                              use_augmentation=apply_augmentation,
                              batch_size=self.training_params.batch_size,
                              batch_size_val=self.training_params.test_batch_size,
                              devices=self.devices,
                              shuffle_train=self.training_params.shuffle_train,
                              skip_images=self.training_params.skip_images)

        # Get more data params from the model params
        self.data_params.from_model(self.model_params)

    def train_loop(self, stage, eval_batchnorm=False):
        """
            Does a training loop for a predefined number of epochs (see training_params.json) using self.training_set

            Args:
                stage: str
                eval_batchnorm: bool
                    if True, batchnorm running statistics will be frozen and not be updated during (second stage)
                    training

            Returns:
                interrupt_code: bool
                    if True, the training loop was interrupted by the user
        """
        self._check_stage(stage)

        optimizer = self._init_optimizer(stage=stage)
        interrupt_code = False
        step_number = 1
        two_d_repr = UmapReduction()  # Object to be used for 2D UMAP representation of embedding
        plot_rul = False

        self.encoder.train()  # Encoder should be in train mode in both stage, torch.no_grad and eval batchnorm will
        # avoid weight updates and running stats updates (if required) respectively:
        if stage == "s2":
            plot_rul = True
            self.predictor.train()
            if eval_batchnorm:
                self.encoder.eval_batchnorm()

        # Epoch loop
        if self.training_params.num_epochs[stage] < 1:
            raise ValueError(f"There should be at least 1 epoch to initiate the training loop")
        for epoch in range(self.training_params.num_epochs[stage]):
            print(f"Epoch {epoch + 1} / {self.training_params.num_epochs[stage]} starts")
            # Break outer loop if interrupt key has been pressed
            # Check for key pressed is inside enumerate(self.training_set) loop
            if interrupt_code:
                break

            if stage == "s2":
                self.encoder.change_dropout(dropout=self.training_params.dropout["s2"])

            average_loss = 0
            n_trains = 1
            # _umap are lists in which to save all relevant data to generate UMAP representation
            embeddings_umap = []
            target_umap = []
            experiments_umap = []

            # Training set loop
            for i, (inputs, targets, times, experiments) in enumerate(tqdm(self.training_set)):
                # Will break the training set loop and epoch loop if interrupt_code_key is pressed
                if keyboard.is_pressed(self.interrupt_code_key):
                    interrupt_code = True
                    break

                optimizer.zero_grad()

                # Normalize inputs
                inputs, targets = inputs.to(self.devices.device_model).to(
                    torch.float32) / self.data_params.normalize_value_dataset, targets.to(self.devices.device_model)
                inputs_no_norm = inputs
                inputs = (inputs - self.data_params.mean) / self.data_params.std

                # Forward propagation and loss calculation
                if stage == "s1":
                    output_transformer, attn_ = self.encoder(inputs)
                    loss = self.loss[stage](output_transformer, targets)
                else:  # if stage is "s2" (guaranteed by check_stage function)
                    with torch.no_grad():
                        output_transformer, attn_ = self.encoder(inputs)
                    predicted = self.predictor(output_transformer)
                    loss = self.loss[stage](predicted, targets)

                # Backward propagation and weight updates
                loss.backward()
                optimizer.step()

                # Relevant parameters for training loop loss calculation
                n_trains = n_trains + 1
                step_number += 1
                average_loss = average_loss + loss

                if stage == "s1":
                    # For 2D representation
                    target_umap.append(targets.detach().cpu())
                    embeddings_umap.append(output_transformer.detach().cpu())
                    experiments_umap.append(experiments.detach().cpu())

                    # To display (example) attention weights (displayed once to tensorboard at the start of each epoch)
                    if i == 0:
                        attn_ = attention.Attn(
                            attn_weights=attn_,
                            n_patches_height=self.data_params.n_patches_height,
                            n_patches_width=self.data_params.n_patches_width,
                            patch_height=self.model_params.patch_height,
                            patch_width=self.model_params.patch_width,
                            normalize_value_dataset=self.data_params.normalize_value_dataset,
                        )
                        attn_.spat_attn_to_writer(writer=self.writer, inputs=inputs_no_norm, epoch=epoch)

            # Display 2D representation at end of each epoch
            if stage == "s1":
                print("Creating UMAP representation")
                two_d_repr.plot_results(
                    inputs=torch.cat(embeddings_umap),
                    targets=torch.cat(target_umap),
                    experiments=torch.cat(experiments_umap),
                    writer=self.writer,
                    epoch=epoch,
                    type_eval="training",
                )

            # Training set loss to tensorboard
            training_loss = float(average_loss / n_trains)
            self.writer.add_scalar(f"loss_stage_{stage[-1]}/training", training_loss, epoch + 1)
            self.losses[stage]["training"][epoch] = training_loss

            # Non-augmented training set loss to tensorboard
            training_eval_loss = self.evaluate(dataset=self.training_set_eval, stage=stage, type_eval="training eval",
                                               plot_rul=plot_rul, epoch=epoch, plot_umap=True)
            self.writer.add_scalar(f"loss_stage_{stage[-1]}/training_eval", training_eval_loss, epoch + 1)

            # Validation set loss to tensorboard
            if self.data_params.validation_set_exists:
                validation_loss = self.evaluate(dataset=self.validation_set, stage=stage, type_eval="validation",
                                                plot_rul=plot_rul, epoch=epoch, plot_umap=True)
                self.writer.add_scalar(f"loss_stage_{stage[-1]}/validation", validation_loss, epoch + 1)

            # Testing set loss to tensorboard
            if self.data_params.testing_set_exists:
                testing_loss = self.evaluate(dataset=self.testing_set, stage=stage, type_eval="testing",
                                             plot_rul=plot_rul, epoch=epoch, plot_umap=True)
                self.writer.add_scalar(f"loss_stage_{stage[-1]}/testing", testing_loss, epoch + 1)

            # Print losses to user
            print("Training       Training eval", end="")
            if self.data_params.validation_set_exists:
                print("   Validation", end="")
            if self.data_params.testing_set_exists:
                print("      Testing")
            else:
                print("\n")
            print(f"{'{:.3f}'.format(round(training_loss, 3))}"
                  f"{'          {:.3f}'.format(round(training_eval_loss, 3))}", end="")
            if self.data_params.validation_set_exists:
                print(f"           {'{:.3f}'.format(round(validation_loss, 3))}", end="")
            if self.data_params.testing_set_exists:
                print(f"           {'{:.3f}'.format(round(testing_loss, 3))}")
            else:
                print("\n")

            # Save model every epoch
            if stage == "s1":
                torch.save(self.encoder.state_dict(),
                           f"{self.logdir[stage]}/encoder_state_dict_epoch_{epoch + 1}.pt")
            else:
                torch.save(self.encoder.state_dict(),
                           f"{self.logdir[stage]}/encoder_state_dict_epoch_bn_{epoch + 1}.pt")
                torch.save(self.predictor.state_dict(),
                           f"{self.logdir[stage]}/predictor_state_dict_epoch_{epoch + 1}.pt")

        # Determine the best epoch
        if stage == "s1":
            best_epoch = self._select_best(stage)
            torch.save(torch.load(f"{self.logdir[stage]}/"
                                  f"encoder_state_dict_epoch_{best_epoch + 1}.pt"),
                       f"{self.logdir[stage]}/encoder_state_dict_epoch_best.pt")
        else:
            best_epoch = self._select_best(stage)
            torch.save(torch.load(f"{self.logdir[stage]}/"
                                  f"predictor_state_dict_epoch_{best_epoch + 1}.pt"),
                       f"{self.logdir[stage]}/predictor_state_dict_epoch_best.pt")

            # if eval batchnorm was false, encoder with new running stats has to be saved as well:
            if not eval_batchnorm:
                torch.save(torch.load(f"{self.logdir[stage]}/"
                                      f"encoder_state_dict_epoch_bn_{best_epoch + 1}.pt"),
                           f"{self.logdir[stage]}/encoder_state_dict_epoch_best_bn.pt")

        return interrupt_code

    def evaluate(self, dataset, stage, type_eval, epoch=None, mc_samples=5, plot_rul=False, split_rul=False,
                 plot_umap=False, show=False, eval_mode=True, export_attn=False):
        """
            Forward propagates an entire dataset based on current encoder and/or predictor state to evaluate the
            (current state) of the model

            Args:
                dataset: data.BuildDataset
                stage: str
                type_eval: str
                    type of dataset to evaluate (e.g. training, testing, validation...)
                epoch: int
                    current epoch (start counting from 0)
                mc_samples: int
                    number of Monte Carlo samples to use for MC dropout
                plot_rul: bool
                    if True, it will plot the predicted RUL to tensorboard (only relevant for stage "s2")
                split_rul: bool
                    if True, it will plot all RULs for each experiment in a different figure, otherwise all in same fig
                plot_umap: bool
                    if True, it will plot the 2D UMAP representation to tensorboard
                show: bool
                    if True, it will also show the RUL and UMAP representation (essentially executes plt.show() at end)
                eval_mode: bool
                    if True, the model is set to evaluation mode, otherwise in training mode (to allow dropout)
                    torch.no_grad() is applied in any case, so gradients are not computed regardless of this arg
                export_attn: bool
                    if True, the current evaluation is meant to only export attention weights, dropout is set to 0.0

            Returns:
                test_loss: float
                    average loss for entire dataset
        """
        # Check type_eval
        if type_eval not in ["training", "training eval", "validation", "testing"]:
            raise ValueError(f"{type_eval} is not a recognized type of eval mode\n"
                             f"type_eval should be one of 'training', 'training eval', 'validation', 'testing'")

        two_d_repr = UmapReduction()

        if stage == "s1":
            mc_samples = 1  # First stage is evaluated deterministic (dropout is set to 0.0 for eval only)

        # Put all models/layers in the required mode for training
        if eval_mode:
            if self.encoder is not None:
                self.encoder.eval()
        else:
            # Batchnorm running statistics should never be updated during testing
            self.encoder.eval_batchnorm()

        # Predictor has no dropout, so should always be eval
        if self.predictor is not None:
            self.predictor.eval()

        with torch.no_grad():  # Avoids any weight updates/data leakage (note that batchnorm is still set to eval mode
            # seperately to avoid updating the running stats)

            if stage == "s2" and not export_attn:
                self.encoder.change_dropout(dropout=self.training_params.dropout["s1"], mc_dropout=True)

            if export_attn:
                mc_samples = 1  # Dropout is set to zero, so no need to produced more samples as it is deterministic
                self.encoder.change_dropout(dropout=0.0, mc_dropout=False)

            # Params for loss calculation
            n_tests = 0
            test_loss = np.array([])
            # Params to plot RUL
            rul_pred = []
            rul_targ = []
            rul_pred_low = []
            rul_pred_high = []
            experiments_list = []
            times_list = []
            # Params to plot umap
            embeddings_umap = []
            target_umap = []
            experiments_umap = []

            # Iterate through dataset to evaluate
            for i, (inputs, targets, times, experiments) in enumerate(dataset):
                if i == 0:
                    first_batch_size = inputs.shape[0]
                last_batch_size = inputs.shape[0]

                # Normalize inputs
                inputs, targets, times = inputs.to(self.devices.device_model).to(
                    torch.float32) / self.data_params.normalize_value_dataset, targets.to(
                    self.devices.device_model), times.to(self.devices.device_model)
                inputs = (inputs - self.data_params.mean) / self.data_params.std

                # Forward propagation and loss calculation
                loss = 0
                mc_predicted = []
                for _ in range(mc_samples):
                    output_transformer, attn_ = self.encoder(inputs)
                    if stage == "s2":
                        predicted = self.predictor(output_transformer)
                    else:  # only save loss in first stage (second stage loss calculation happens after CI calculation)
                        # assumes mc_samples == 1
                        loss = self.loss[stage](output_transformer, targets)
                    if stage == "s2":
                        mc_predicted.append(predicted)

                # Calculate confidence intervals for RUL
                if stage == "s2" and not export_attn:
                    # Stack RULs for each respective timestep on dim=-1
                    mc_predicted = torch.stack(mc_predicted, dim=-1).squeeze()
                    # De-standardize RULs
                    mc_predicted = mc_predicted * self.data_params.max_train + self.data_params.min_train
                    # Loop through all RUL timesteps to calculate RUL for each timestep individually
                    predicted = []
                    predicted_low = []
                    predicted_high = []
                    for j in range(mc_predicted.shape[0]):
                        CI, pred_mean = utils.confidence_intervals(mc_predicted[j, :].detach().cpu().numpy(), 0.95)
                        predicted.append(pred_mean)
                        predicted_low.append(CI.low)
                        predicted_high.append(CI.high)
                    # Save results to tensors
                    predicted = torch.from_numpy(np.array(predicted)).to(self.devices.device_model)
                    predicted_low = torch.from_numpy(np.array(predicted_low)).to(self.devices.device_model)
                    predicted_high = torch.from_numpy(np.array(predicted_high)).to(self.devices.device_model)
                    # Calculate losses with predicted mean of RUL
                    loss = loss + self.loss[stage](
                        (predicted - self.data_params.min_train) / self.data_params.max_train,
                        targets.squeeze())

                # Export attention weights to
                elif stage == "s2" and export_attn:
                    attn_ = attention.Attn(
                        attn_weights=attn_,
                        n_patches_height=self.data_params.n_patches_height,
                        n_patches_width=self.data_params.n_patches_width,
                        patch_height=self.model_params.patch_height,
                        patch_width=self.model_params.patch_width,
                        normalize_value_dataset=self.data_params.normalize_value_dataset,
                    )
                    attn_.spat_attn_save(inputs=inputs, timestep=1,
                                         folder_name_save=self.logdir["attn"], index=i)
                    attn_.temp_attn_draw(folder_name_save=self.logdir["attn"], index=i)
                    plt.close('all')

                test_loss = np.append(test_loss, float(loss))
                n_tests += 1

                # For 2D representation
                target_umap.append(targets.detach().cpu())
                embeddings_umap.append(output_transformer.detach().cpu())
                experiments_umap.append(experiments.detach().cpu())

                # Save predicted RUL and target RUL for plot
                if plot_rul:
                    for index_predicted in range(len(predicted)):
                        rul_pred.append(float(predicted[index_predicted]))
                        rul_targ.append(float(targets[index_predicted]))
                        rul_pred_low.append(float(predicted_low[index_predicted]))
                        rul_pred_high.append(float(predicted_high[index_predicted]))
                        experiments_list.append(float(experiments[index_predicted]))
                        times_list.append(float(times[index_predicted]))

            # Plot RUL vs time
            if plot_rul:
                if not split_rul:  # Don't split RULs for each view and sample
                    fig, ax = plt.subplots(dpi=200)
                    ax.plot(range(len(rul_targ)), rul_pred, color='blue', label="Predicted values")
                    ax.plot(range(len(rul_targ)), rul_targ, ls="--", color='black', label="True values")
                    ax.plot(
                        range(len(rul_targ)),
                        np.array(rul_pred) - np.array(rul_pred_low),
                        linewidth=0.5,
                        ls="-.",
                        color='blue',
                        label=f"95% Confidence intervals",
                    )
                    ax.plot(
                        range(len(rul_targ)),
                        np.array(rul_pred) + np.array(rul_pred_high),
                        linewidth=0.5,
                        ls="-.",
                        color='blue'
                    )
                    ax.set_xlabel("Total time [s]")
                    ax.set_ylabel("RUL [s]")
                    plt.draw()
                else:  # Split RULs for each experiment and view
                    current_experiment = experiments_list[0]
                    rul_list_pred_plot = []
                    rul_list_pred_plot_low = []
                    rul_list_pred_plot_high = []
                    rul_list_targ_plot = []
                    times_list_plot = []
                    for k in range(len(experiments_list)):  # iterate through all experiments AND views
                        #  Detects when RUL belongs to a different experiment/view:
                        if experiments_list[k] != current_experiment or k == len(experiments_list) - 1:
                            fig, ax = plt.subplots()
                            times_list_plot = np.array(times_list_plot)
                            times_list_plot = times_list_plot - times_list_plot[0]
                            ax.plot(times_list_plot, rul_list_targ_plot, ls="--", color='black', label="True values")
                            ax.plot(times_list_plot, rul_list_pred_plot, color='blue', label="Predicted values")

                            ax.plot(
                                times_list_plot,
                                np.array(rul_list_pred_plot) - np.array(rul_list_pred_plot_low),
                                linewidth=0.5,
                                alpha=0.3,
                                ls="-.",
                                color='blue',
                                label=f"95% Confidence intervals",
                            )
                            ax.plot(
                                times_list_plot,
                                np.array(rul_list_pred_plot) + np.array(rul_list_pred_plot_high),
                                linewidth=0.5,
                                alpha=0.3,
                                ls="-.",
                                color='blue'
                            )

                            ax.set_xlabel("Time [s]")
                            ax.set_ylabel("RUL [s]")
                            rul_list_pred_plot = (torch.Tensor(
                                rul_list_pred_plot) - self.data_params.min_train) / self.data_params.max_train
                            rul_list_targ_plot = (torch.Tensor(
                                rul_list_targ_plot) - self.data_params.min_train) / self.data_params.max_train
                            error = float(self.loss[stage](rul_list_pred_plot, rul_list_targ_plot))
                            ax.scatter([], [], label=f"$MAE$ = {round(error, 3)}", alpha=0)  # dummy label for legend
                            ax.legend()
                            plt.draw()
                            plt.savefig(f"{self.logdir['rul_figures']}/RUL_prediction_experiment_"
                                        f"{int(current_experiment // 2)}"
                                        f"_view_{int(current_experiment % 2)}_{type_eval}.pdf")

                            # Reset current experiment identifier and all lists
                            current_experiment = experiments_list[k]
                            rul_list_pred_plot = []
                            rul_list_targ_plot = []
                            rul_list_pred_plot_low = []
                            rul_list_pred_plot_high = []
                            times_list_plot = []

                        # Append RULs at each timestep
                        rul_list_targ_plot.append(rul_targ[k] * self.data_params.max_train + self.data_params.min_train)
                        rul_list_pred_plot.append(rul_pred[k])
                        rul_list_pred_plot_low.append(rul_pred_low[k])
                        rul_list_pred_plot_high.append(rul_pred_high[k])
                        times_list_plot.append(times_list[k])

            # Plot 2D UMAP representation
            if plot_umap:
                two_d_repr.plot_results(
                    inputs=torch.cat(embeddings_umap),
                    targets=torch.cat(target_umap),
                    experiments=torch.cat(experiments_umap),
                    writer=self.writer,
                    epoch=epoch,
                    type_eval=type_eval.replace(" ", "_"),
                    export=self.logdir["umap"],
                )

            # Show plots or close them (to avoid memory leak in matplotlib)
            if show:
                plt.show()
            else:
                plt.close('all')

        # Makes sure last batch (with size <= batch size) is not overrepresented when calculating average loss
        test_loss = weight_loss(test_loss, last_batch_size, first_batch_size)

        # Put the models back in the original mode they were in
        if self.encoder is not None:
            self.encoder.train()
            if stage == "s2":
                self.encoder.train_batchnorm()
        if self.predictor is not None:
            self.predictor.train()

        # Save losses to the losses dataframe
        if epoch is not None:
            self.losses[stage][type_eval][epoch] = test_loss
        return test_loss

    def train_two_stage(self):
        """
            Does two-stage training as described in paper
        """
        self._init_folder()

        # First stage training
        stage = "s1"
        print("\nInitiating first stage training")
        self.load_dataset(self.folder_name_data, self.apply_augmentation)
        self.encoder = self._init_encoder(stage=stage)
        interrupt_code = self.train_loop(stage=stage)
        self.losses[stage].to_csv(self.logdir["0"] + f"/losses_{stage}.csv")

        # Check if user wants to continue training if user interrupted code during first stage training
        if interrupt_code:
            warnings.warn("You have interrupted the first stage training, "
                          "do you wish to continue to second stage training? (y/n)",
                          stacklevel=2)
            continue_stage_2 = input()
            if continue_stage_2 == "y":
                stage = "s2"
            else:
                stage = "stop"
                print("Stopping two stage training")
        else:
            stage = "s2"

        # Second stage training
        if stage == "s2":
            print("\nInitiating second stage training")
            self.load_dataset(self.folder_name_data, False)
            self.encoder = self._init_encoder(stage=stage,
                                              model_path=f"{self.logdir['s1']}/"
                                                         f"encoder_state_dict_epoch_best.pt")
            self.predictor = self._init_predictor()

            interrupt_code = self.train_loop(stage=stage)
            if interrupt_code:
                warnings.warn("You have interrupted the second stage training",
                              stacklevel=2)
            self.losses[stage].to_csv(self.logdir["0"] + f"/losses_{stage}.csv")

            self.save_params()

    def train_stage_2(self, folder_name):
        """
            In case the first stage training got interrupted, and you (accidentally) did not continue for second stage
            training, this function can be used to train second stage without having to redo the
            (computational intensive) first stage training

            Args:
                folder_name: str
                    name of folder in runs directory from which to load first stage training data
        """
        print("\nInitiating second stage training")
        self._init_folder(folder_name)
        self.load_dataset(self.folder_name_data, False)

        self.encoder = self._init_encoder(stage="s2",
                                          model_path=f"{self.logdir['s1']}/"
                                                     f"encoder_state_dict_epoch_best.pt")
        self.predictor = self._init_predictor()

        interrupt_code = self.train_loop(stage="s2")
        if interrupt_code:
            warnings.warn("You have interrupted the second stage training",
                          stacklevel=2)
        self.losses["s2"].to_csv(self.logdir["0"] + f"/losses_{'s2'}.csv")

        self.save_params()

    def test(self, folder_name, mc_samples_test=40, export_attention_weights=False, export_umap=False):
        """
            Tests the model in runs/folder_name directory, printing losses and exporting several plots depending on args

            Args:
                folder_name: str
                    name of folder in runs directory from which to load first stage training data
                mc_samples_test: int
                    number of Monte Carlo samples to use for MC dropout
                export_attention_weights: bool
                    if True, exports attention weights (with dropout = 0.0 and consequently no MC dropout)
                export_umap: bool
                    if True, exports UMAP plots
        """

        print("\nInitiating testing phase")
        self._init_folder(folder_name)

        if export_umap:
            self.load_dataset(self.folder_name_data, True)
        else:
            self.load_dataset(self.folder_name_data, False)

        self.encoder = self._init_encoder(stage="s2",
                                          model_path=f"{self.logdir['s2']}/"
                                                     f"encoder_state_dict_epoch_best_bn.pt")
        self.predictor = self._init_predictor(model_path=f"{self.logdir['s2']}/predictor_state_dict_epoch_best.pt")

        if export_umap:
            print("Exporting UMAP representation, this might take up to a few minutes")
            self.evaluate(dataset=self.training_set_eval, stage="s1", type_eval="training eval",
                          plot_rul=False, plot_umap=True, mc_samples=1, export_attn=False)

            self.evaluate(dataset=self.training_set, stage="s1", type_eval="training",
                          plot_rul=False, plot_umap=True, mc_samples=1, export_attn=False, eval_mode=False)
            self.load_dataset(self.folder_name_data, False)

        stage = "s2"
        plot_rul = True
        split_rul = True

        # Export losses
        self._export_losses()

        # Export attention weights
        if self.data_params.testing_set_exists and export_attention_weights:
            self.evaluate(dataset=self.testing_set, stage=stage, type_eval="testing",
                          plot_rul=False, split_rul=split_rul, plot_umap=False,
                          mc_samples=1, export_attn=True)

        # Non-augmented training set loss to tensorboard
        training_eval_loss = self.evaluate(dataset=self.training_set_eval, stage=stage, type_eval="training eval",
                                           plot_rul=plot_rul, split_rul=split_rul, plot_umap=False,
                                           mc_samples=mc_samples_test)

        # Validation set loss to tensorboard
        if self.data_params.validation_set_exists:
            validation_loss = self.evaluate(dataset=self.validation_set, stage=stage, type_eval="validation",
                                            plot_rul=plot_rul, split_rul=split_rul, plot_umap=False,
                                            mc_samples=mc_samples_test)

        # Testing set loss to tensorboard
        if self.data_params.testing_set_exists:
            testing_loss = self.evaluate(dataset=self.testing_set, stage=stage, type_eval="testing",
                                         plot_rul=plot_rul, split_rul=split_rul, plot_umap=False,
                                         mc_samples=mc_samples_test)

        # Print losses to user
        print(f"Losses:")
        print("Training eval", end="")
        if self.data_params.validation_set_exists:
            print("   Validation", end="")
        if self.data_params.testing_set_exists:
            print("      Testing")
        else:
            print("\n")
        print(f"{'          {:.3f}'.format(round(training_eval_loss, 3))}", end="")
        if self.data_params.validation_set_exists:
            print(f"           {'{:.3f}'.format(round(validation_loss, 3))}", end="")
        if self.data_params.testing_set_exists:
            print(f"           {'{:.3f}'.format(round(testing_loss, 3))}")
        else:
            print("\n")
