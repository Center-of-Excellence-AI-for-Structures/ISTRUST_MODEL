import umap
import matplotlib.pyplot as plt
import utils
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings


class UmapReduction(umap.UMAP):
    def __init__(self):
        super().__init__()

    def plot_results(self, inputs, writer=None, epoch=None, targets=None, experiments=None, type_eval='unknown_type',
                     export=None):
        """
            Plot the results of UMAP reduction, also does the actual umap reduction
            Source for discrete colorbar: https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar

            Args:
                inputs: torch.Tensor
                    embeddings originating from the encoder, shape should be [batch_size, d_model]
                writer: tensorboard writer
                    if not None, saves the UMAP representation to tensorboard writer given
                epoch: int
                    current epoch, start count from 0
                targets: torch.Tensor
                    targets for the RULs corresponding to the embeddings, shape should be [batch_size, 1]
                experiments: torch.Tensor
                    experiment identifiers corresponding to the embeddings, shape should be [batch_size, 1]
                type_eval: str
                    type of dataset to evaluate (e.g. training, testing, validation...)
                export: str
                    if not None, should be equal to folder to which to export a pdf figure of the UMAP,
                    otherwise it is only saved to tensorboard (if writer is not None)

        """
        # Check type_eval
        if type_eval not in ["unknown_type", "training", "training_eval", "validation", "testing"]:
            raise ValueError(f"{type_eval} is not a recognized type of eval mode\n"
                             f"type_eval should be one of 'unknown_type', 'training', 'training_eval', 'validation', "
                             f"'testing'")

        # Actual reduction to UMAP representation
        embedding = self.fit_transform(inputs)

        # Generate first plot with targets and thus continuous colors
        fig, ax = plt.subplots(1, 2, dpi=180, figsize=(7, 3), squeeze=True)
        if targets is not None:  # if there are target values, there should be a third dimension: colour
            im_tar = ax[0].scatter(embedding[:, 0], embedding[:, 1], c=targets, cmap='plasma', s=5)
            im_tar.set_rasterized(True)
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[0].set_box_aspect(1)
            ax[0].set_xlabel("Primary component [arbitrary units]")
            ax[0].set_ylabel("Secondary component [arbitrary units]")
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_tar, cax=cax)
        else:  # otherwise only primary and secondary components are plotted
            ax[0].scatter(embedding[:, 0], embedding[:, 1], s=5)
            ax[0].set_xlabel("Primary component [arbitrary units]")
            ax[0].set_ylabel("Secondary component [arbitrary units]")

        # Generate second plot with experiments and thus discrete colors
        if experiments is not None:
            experiments = (experiments // 2).to(torch.int32)
            # get discrete colormap
            cmap = plt.get_cmap("Set1", int(torch.max(experiments)) - int(torch.min(experiments)) + 1)
            # set limits .5 outside true range
            im_exp = ax[1].scatter(embedding[:, 0], embedding[:, 1], s=5, c=experiments, cmap=cmap,
                                   vmin=int(torch.min(experiments)) - 0.5, vmax=int(torch.max(experiments)) + 0.5)
            im_exp.set_rasterized(True)
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[1].set_box_aspect(1)
            ax[1].set_xlabel("Primary component [arbitrary units]")
            ax[1].set_ylabel("Secondary component [arbitrary units]")
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_exp, cax=cax,
                         ticks=torch.arange(int(torch.min(experiments)), int(torch.max(experiments)) + 1))
        fig.tight_layout()
        if export is not None:
            fig.savefig(f"{export}/{type_eval}_umap_repr_best.pdf", dpi=500)

        # Plot to tensorboard
        if writer is not None:
            if epoch is None:
                warnings.warn("Epoch was not specified, defaulting to epoch=0")
                epoch = 0
            utils.plot_to_tensorboard(writer, fig, f'2D_representation/{type_eval}', global_step=epoch + 1)
