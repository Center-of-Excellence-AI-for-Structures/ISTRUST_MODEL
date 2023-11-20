import torch
import matplotlib.pyplot as plt
import torchvision
import utils


def attn_self_token_merger(attn_weights, keep_heads=False):
    """
        Converts a list of two self-attention weights and token-attention weights to a single tensor of interpretable
        attention weights, as described by equation 9 (A = A_self * A_token = ...) in the paper

        Args:
            attn_weights: list(tensor, tensor), shape of tensors: [*, n_queries, n_heads, n_keys]
                A list of the tensor containing self-attention and token-attention weights respectively
            keep_heads: bool
                if False, the attention weights are averaged over the heads dimension before matmul is applied

        Returns:
            merged attention weights: tensor, shape: [*, n_queries_token, n_keys_self]
    """
    # keep_heads == True does not make sense for our model architecture (head 1 of self-attention layer has no
    # correlation with head 1 of token-attention layer)
    if keep_heads:
        raise NotImplementedError("keep_heads is currently not supported due to the model architecture")
    else:
        self_attn, token_attn = attn_weights
        self_attn, token_attn = torch.mean(self_attn, dim=2), torch.mean(token_attn, dim=2)  # average over heads dim
    return torch.bmm(token_attn, self_attn)


def compute_reduced_attn(type_attn):
    """
        Reduces attention weights over head dimension, and merges self and token attention weights if applicable

        Args:
            type_attn: list(tensor, tensor) OR tensor OR None
                attention weights

        Returns:
            reduced attention weights: tensor
    """
    if type_attn is not None:
        if type(type_attn) == list:
            return attn_self_token_merger(type_attn)
        else:
            return torch.mean(type_attn, dim=-2)
    else:
        return None


class Attn:
    """
        Class that handles visualisation of attention weights

        Parameters:
            attn_weights: dict: {"spat_attn": tensor, "temp_attn": tensor}, shapes: [*, n_queries, n_heads, n_keys]
                dictionary containing spatial and temporal attention weights as tensors
            n_patches_height: int
            n_patches_width: int
            patch_height: int
            patch_width: int
            normalize_value_dataset: int
                maximum value to which the images shall be normalized, typically 255
    """
    def __init__(self, attn_weights, n_patches_height, n_patches_width, patch_height, patch_width,
                 normalize_value_dataset):
        self.attn_spat = attn_weights["spat_attn"]
        self.attn_temp = attn_weights["temp_attn"]
        self.n_patches_height = n_patches_height
        self.n_patches_width = n_patches_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.normalize_value_dataset = normalize_value_dataset
        self.attn_temp_reduced = compute_reduced_attn(self.attn_temp)
        self.attn_spat_reduced = compute_reduced_attn(self.attn_spat)
        # de-split batch dimensions and spatial dimension
        self.attn_temp_reduced = torch.reshape(self.attn_temp_reduced,
                                               [-1, self.n_patches_width * self.n_patches_height,
                                                self.attn_temp_reduced.shape[-2], self.attn_temp_reduced.shape[-1]])
        # assume every temporal part can be averaged over spatial dimensions
        self.attn_temp_reduced_avg = torch.mean(self.attn_temp_reduced, dim=-3)
        # OR weight each temporal part taking into account the spatial relevance (and thus spatial attention weights)
        self.attn_temp_reduced_weighted = \
            torch.sum(self.attn_temp_reduced.squeeze() * self.attn_spat_reduced.swapdims(-1, -2), dim=-2)

    def _spat_attn_unflatten(self, repeat=True):
        """
            Returns spatial attention weights with unflattened spatial dimension

            Args:
                self.attn_spat_reduced: tensor, shape: [*, n_images, n_patches]
                repeat: bool
                    determines return shape, see below

            Returns:
                spat_attn: tensor, shape:
                    if repeat:
                        [*, n_images, image_height, image_width]
                    else:
                        [*, n_images, n_patches_height, n_patches_width]
        """
        spat_attn = torch.reshape(self.attn_spat_reduced,
                                  [self.attn_spat_reduced.shape[0], self.n_patches_height, self.n_patches_width])
        if repeat:
            spat_attn = spat_attn.repeat_interleave(int(self.patch_height), dim=-2)\
                .repeat_interleave(int(self.patch_width), dim=-1)
        return spat_attn

    def _spat_attn_to_image(self, inputs):
        """
            Merges input image and attention weights where the brightness of a patch in the output image represents
            the relative attention

            Args:
                inputs: tensor, shape: [batch, temp, image_height, image_width]
        """
        spat_attn = self._spat_attn_unflatten()
        # Normalize attention weights between 0 and 1
        spat_attn_plot = (self.normalize_value_dataset * spat_attn / torch.max(spat_attn)).detach().cpu().to(
            torch.uint8)

        # Normalize image pixels between 0 and self.normalize_value_dataset
        image = (self.normalize_value_dataset * torch.squeeze(inputs, dim=-3) / torch.max(
            torch.squeeze(inputs, dim=-3))).detach().cpu().to(torch.uint8)
        # Take last image for spatial attention weights
        image = image[:, -1, :, :].unsqueeze(dim=1)
        spat_attn_plot = spat_attn_plot.reshape(*image.shape)

        # merge the attention weights and input image
        attended_image = spat_attn_plot.to(torch.int32) * image.to(torch.int32)
        attended_image = (self.normalize_value_dataset * attended_image / torch.max(attended_image)).to(torch.uint8)
        # Put the input image, spatial attention weights and merged image next to each other
        return torch.cat((image, spat_attn_plot, attended_image), dim=-1)

    def spat_attn_to_writer(self, writer, inputs, epoch):
        """
            Plots the spatial attention weights from self._spat_attn_to_image to tensorboard

            Args:
                writer: tensorboard writer
                    tensorboard writer object to plot to
                inputs: tensor, shape: [batch, temp, image_height, image_width]
                    input images
                epoch: int
                    epoch, counting should have started from 0
        """
        final_image = self._spat_attn_to_image(inputs=inputs)
        # Generate plot for each image in the batch
        for batch in range(final_image.shape[0]):
            current_image = final_image[batch, -1]
            final_image_grid = torchvision.utils.make_grid(current_image)
            plt.imshow(current_image, cmap="Greys")
            writer.add_image(f"spatial_attention/epoch_{epoch + 1}", final_image_grid, global_step=batch)

    def spat_attn_draw(self, inputs):
        """
            Draws the spatial attention weights from self._spat_attn_to_image in matplotlib, does not show!

            Args:
                inputs: tensor, shape: [batch, temp, image_height, image_width]
                    input images
        """
        final_image = self._spat_attn_to_image(inputs=inputs)
        # Generate plot for each image in the batch
        for batch in range(final_image.shape[0]):
            current_image = final_image[batch, -1]
            plt.figure()
            plt.title(f"Batch {batch}")
            plt.imshow(current_image)
            plt.draw()

    def spat_attn_save(self, inputs, timestep, folder_name_save, index):
        """
            Saves both the attention weights and input images, to be used for the paper

            Args:
                inputs: tensor, shape: [batch, temp, image_height, image_width]
                    input images
                timestep: int
                    timestep of the specimen (so not the timestep in the input images themselves)
                folder_name_save: str
                    folder path from root in which to save attention weights (typically runs/.../attention)
                index: int
                    index used to save model, typically index of batch in a dataset (for i, _ in enumerate(dataset))
        """
        if self.attn_spat_reduced is not None:
            spat_attn = self._spat_attn_unflatten()
            # Normalize attention weights between 0 and 1
            spat_attn_plot = (self.normalize_value_dataset * spat_attn / torch.max(spat_attn)).detach().cpu()
            # Normalize image pixels between 0 and self.normalize_value_dataset
            image = (self.normalize_value_dataset * torch.squeeze(inputs, dim=-3) /
                     torch.max(torch.squeeze(inputs, dim=-3))).detach().cpu()
            # Take last image for spatial attention weights
            image = image[:, -1, :, :].unsqueeze(dim=1)
            # Generate plot for each image in the batch
            for batch in range(0, image.shape[0], timestep):
                current_image = image[batch, -1]
                # plot the last input image
                plt.figure(figsize=(8, 16))
                plt.axis("off")
                current_attention = spat_attn_plot[batch]
                plt.imshow(current_image, cmap="gray")
                plt.tight_layout()
                plt.draw()
                plt.savefig(f"{folder_name_save}/img_{index}_{batch}.pdf")
                # plot the spatial attention weights
                plt.figure(figsize=(8, 16))
                plt.axis("off")
                plt.imshow(current_attention/torch.sum(current_attention), cmap="magma")
                plt.tight_layout()
                plt.draw()
                plt.savefig(f"{folder_name_save}/spat_attn_{index}_{batch}.pdf")

    def temp_attn_draw(self, writer=None, epoch=None, folder_name_save=None, index=None):
        """
            Saves both the attention weights and input images, to be used for the paper

            Args:
                writer: tensorboard writer
                    tensorboard writer object to plot to
                epoch: int
                    epoch, counting should have started from 0
                folder_name_save: str
                    folder path from root in which to save attention weights (typically runs/.../attention)
                index: int
                    index used to save model, typically index of batch in a dataset (for i, _ in enumerate(dataset))
        """
        if self.attn_temp_reduced_avg is not None:
            for batch in range(self.attn_temp_reduced_weighted.shape[0]):
                # Plot to matplotlib
                fig, ax = plt.subplots()
                ax.set_xticks(list(range(1, 1 + len(self.attn_temp_reduced_weighted[batch].squeeze()))))
                ax.bar(range(1, 1 + len(self.attn_temp_reduced_weighted[batch].squeeze())),
                       self.attn_temp_reduced_weighted[batch].squeeze().detach().cpu())
                plt.draw()
                # Plot to tensorboard
                if (writer is not None) and (epoch is not None):
                    utils.plot_to_tensorboard(writer, fig, f"temporal_attention_epoch_{epoch}_batch_{batch}")
                if folder_name_save is not None:
                    plt.savefig(f"{folder_name_save}/temp_attn_{index}_{batch}.pdf")
