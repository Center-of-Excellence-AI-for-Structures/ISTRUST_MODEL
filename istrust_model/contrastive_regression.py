import torch


class SupCR(torch.nn.Module):
    """
        Supervised contrastive regression loss
        Loss function be used to train any encoder without the final layer, in this case the predictor

        Parameters:
            temp: float
                softmax temperature of the loss function (for more info see equation 10 in the paper)
    """
    def __init__(self, temp):
        super().__init__()
        self.temp_softmax = temp

    @staticmethod
    def _l2_norm(x, y):
        return torch.linalg.vector_norm(x - y, ord=2, dim=-1)

    @staticmethod
    def _mae(x, y):
        return torch.abs(x - y)

    def _exp_sim(self, x, y):
        """
            Computes the exponential of the similarity function between the two input tensors

            Expected input shape: [*, d_embedded], where d_embedded is typically = d_model
        """
        return torch.exp(-self._l2_norm(x, y) / self.temp_softmax)

    def _label_dist(self, targets):
        """
            Computes the label distance between the targets/labels
        """
        sim_targets = self._mae(targets, targets.transpose(0, 1))
        identity = torch.ones_like(sim_targets, device=targets.device) - \
                   torch.eye(sim_targets.shape[0], device=targets.device)
        identity = identity.unsqueeze(-2) * torch.ones(identity.shape[-1], device=targets.device).unsqueeze(0).\
            unsqueeze(-1)
        sim_targets = (torch.repeat_interleave(sim_targets.unsqueeze(-2), sim_targets.shape[0], dim=-2) >=
                       torch.repeat_interleave(sim_targets.unsqueeze(-1), sim_targets.shape[0], dim=-1))
        sim_targets = sim_targets*identity
        return sim_targets

    def forward(self, embeddings, targets):
        """
            Computes the actual loss function

            Args:
                embeddings: tensor, shape: [2N, d_model]
                    spatial encoded embeddings
                targets: tensor, shape: [2N, 1]
                    target RULs (hence why it is still supervised ;) )
        """
        two_N = embeddings.shape[0]
        targ_matrix = self._label_dist(targets)
        sim_matrix = self._exp_sim(embeddings.unsqueeze(0), embeddings.unsqueeze(0).transpose(0, 1))
        denominator = targ_matrix * sim_matrix.unsqueeze(-2)
        denominator = torch.sum(denominator, dim=-1)
        loss = torch.log(sim_matrix / denominator)
        mask = torch.ones_like(loss, device=embeddings.device) - torch.eye(loss.shape[0], device=embeddings.device)
        loss = torch.sum(loss * mask)
        return - loss / (two_N * (two_N-1))
