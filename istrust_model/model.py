import torch
import torch.nn as nn
from modules import Patching, PositionalEncoding2D, \
    InterpretableTransformerEncoder, PositionalEncoding1D, ConvPatchEmbed


class Encoder(nn.Module):
    """
        Encoder used to obtain the spatial encoded embeddings

        Parameters:
            data_params: DataParams class as described in data.py
                contains hyperparameters related to the dataset
            model_params: ModelParams class as described in params.py
                contains hyperparameters related to the model
            training_params: TrainingParams class as described in params.py
                contains hyperparameters related to the training of the model
            stage: string
                the training stage for which the encoder shall be used
    """
    def __init__(self, data_params, model_params, training_params, stage):
        super().__init__()
        self.n_patches = data_params.n_patches
        self.window_size = data_params.window_size
        self.patching = Patching(image_width=data_params.image_width,
                                 image_height=data_params.image_height,
                                 patch_width=model_params.patch_width,
                                 patch_height=model_params.patch_height)
        self.conv_embedder = ConvPatchEmbed(image_width=data_params.image_width,
                                            image_height=data_params.image_height,
                                            patch_width=model_params.patch_width,
                                            patch_height=model_params.patch_height,
                                            in_chans=data_params.n_features,
                                            d_model=model_params.d_model,
                                            dropout=training_params.dropout[stage])
        self.temporal_positional_encoding = PositionalEncoding1D(d_model=model_params.d_model,
                                                                 n_sequence=self.window_size)
        self.temporal_transformer_encoder = InterpretableTransformerEncoder(d_model=model_params.d_model,
                                                                            n_head=model_params.n_head_temporal,
                                                                            dropout=training_params.dropout[stage],
                                                                            d_value=model_params.d_value)
        self.spatial_positional_encoding = PositionalEncoding2D(d_model=model_params.d_model)
        self.spatial_transformer_encoder = InterpretableTransformerEncoder(d_model=model_params.d_model,
                                                                           n_head=model_params.n_head_spatial,
                                                                           dropout=training_params.dropout[stage],
                                                                           d_value=model_params.d_value)
        self.scaling_factor_weights = model_params.scaling_factor_weights
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            # Kaiming normal weight initialization for convolutional embedding weights
            if "conv" in name and "0.weight" in name:
                torch.nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
                with torch.no_grad():
                    p.copy_(torch.repeat_interleave(p[:, :, :, 0].unsqueeze(-1), p.shape[-1], dim=-1))
            # Convolutional biases and batchnorm biases initialized to zero
            elif "conv" in name and "bias" in name:
                torch.nn.init.zeros_(p)
            # Batchnorm weights (gamma) are initialized to one
            elif "conv" in name:
                torch.nn.init.ones_(p)
            elif "conv" not in name:
                if ("weight" in name) and ("norm" not in name) and ("lazy_layer" not in name):
                    if "embedding_matrix" in name:
                        torch.nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
                    elif ("v_layer" in name) or ("w_h" in name):
                        torch.nn.init.xavier_uniform_(p, self.scaling_factor_weights)
                    else:
                        torch.nn.init.xavier_uniform_(p)
                elif "bias" in name:
                    torch.nn.init.zeros_(p)
                elif "learnable_token" in name:
                    torch.nn.init.kaiming_normal_(p, a=1, mode='fan_out', nonlinearity='leaky_relu')

    def change_dropout(self, dropout=None, mc_dropout=False):
        """
            Function to change the dropout probability of the model and/or to put dropout in train mode (even when
            the rest of the model is in eval mode)

            Args:
                dropout: int or None
                    desired global dropout probability of entire model
                    if None, this function will not change the dropout probability
                mc_dropout: bool
                    if True, dropout stays active during evaluation according to the theory of Monte Carlo dropout

            Returns:
                None
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Dropout):
                if dropout is not None:
                    module.p = dropout
                if mc_dropout:
                    module.train()

    def eval_batchnorm(self):
        for name, module in self.conv_embedder.proj.named_children():
            for name_sub, module_sub in module.named_children():
                if isinstance(module_sub, nn.BatchNorm2d):
                    module_sub.eval()

    def train_batchnorm(self):
        for name, module in self.conv_embedder.proj.named_children():
            for name_sub, module_sub in module.named_children():
                if isinstance(module_sub, nn.BatchNorm2d):
                    module_sub.train()

    def forward(self, x):
        x = self.patching(x)
        x = self.conv_embedder(x)

        # Spatial part moves to batch, temporal part to the dimension that undergoes the transformer-encoder
        x = torch.movedim(x, -4, -2)
        n, p_h, p_w, t, d = x.shape

        # Flatten n, p_h and p_w to a single batch dimension
        x = torch.flatten(x, 0, 2)

        x = x + self.temporal_positional_encoding(x)
        x_temporal_attended, x_temporal_attention_weights = self.temporal_transformer_encoder(x)

        x = torch.squeeze(x_temporal_attended, dim=-2)
        # Retrieve the spatial dimensions from the batch dimension
        x = torch.reshape(x, [n, p_h, p_w, d])

        x = x + self.spatial_positional_encoding(x)
        # Flatten p_h and p_w to a single spatial dimension
        x = torch.flatten(x, 1, 2)
        x_spatial_attended, x_spatial_attention_weights = self.spatial_transformer_encoder(x)
        x = torch.squeeze(x_spatial_attended, dim=-2)
        return x, {'spat_attn': x_spatial_attention_weights, 'temp_attn': x_temporal_attention_weights}


class Predictor(nn.Module):
    """
        Predictor used to obtain the RUL from the spatial encoded embeddings

        Parameters:
            model_params: ModelParams class as described in params.py
                contains hyperparameters related to the model
    """
    def __init__(self, model_params):
        super().__init__()
        self.fcn = torch.nn.Sequential(nn.Linear(model_params.d_model, model_params.d_predictor), nn.ReLU())

    def forward(self, x):
        return self.fcn(x)
