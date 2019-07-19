
import typing

import torch
from torch import nn
from torch.nn import functional as F

from autoencoders import variational
from autoencoders.dist_parameterisers import nn_paramterised_dists
from autoencoders.dist_parameterisers import shallow_distributions
from autoencoders.dist_parameterisers import base_parameterised_distribution
from autoencoders import logging_tools

from graph_neural_networks.pad_pattern import ggnn_pad
from graph_neural_networks.ggnn_general import ggnn_base
from graph_neural_networks.core import utils
from graph_neural_networks.core import mlp

from . import graph_datastructure

class GrapTopGGNN(nn.Module):

    def __init__(self, mlp_project_up, mlp_gate, mlp_func):
        super().__init__()
        self.mlp_project_up = mlp_project_up  # net that goes from [None, h'] to [None, j] with j>h usually
        self.mlp_gate = mlp_gate  # net that goes from [None, h'] to [None, 1]
        self.mlp_func = mlp_func  # net that goes from [None, j] to [None, q]

    def forward(self, node_features):
        """
        :param node_features: shape is [b, v, h']
        :return: shape [b,q]
        """
        b, v, h = node_features.shape
        proj_up = F.tanh(self.mlp_project_up(node_features.view(-1, h)).view(b,v,-1))  # [b,v,j]
        gate_logit = self.mlp_gate(node_features.view(-1, h)).view(b,v,1)   # [b,v,1]
        gate = torch.sigmoid(gate_logit)

        gated_sum = F.tanh(torch.sum(proj_up * gate, dim=1))  # [b,j]
        result = self.mlp_func(gated_sum)
        return result


class EncoderNet(nn.Module):
    def __init__(self, graph_hidden_layer_size, cuda_details, T, out_dim):
        super().__init__()
        self.hidden_layer_size = graph_hidden_layer_size

        # Node features
        edge_names = [str(x) for x in graph_datastructure.CHEM_DETAILS.bond_types]
        self.ggnn = ggnn_pad.GGNNPad(
            ggnn_base.GGNNParams(graph_hidden_layer_size, edge_names, cuda_details, T))

        # Aggregation function
        mlp_project_up = mlp.MLP(mlp.MlpParams(graph_hidden_layer_size, 128, []))
        mlp_gate = mlp.MLP(mlp.MlpParams(graph_hidden_layer_size, 1, []))
        mlp_down = mlp.MLP(mlp.MlpParams(128, out_dim, []))
        self.ggnn_top = GrapTopGGNN(mlp_project_up, mlp_gate, mlp_down)

    def forward(self, graph: graph_datastructure.OneHotMolecularGraphs):
        """
        :param adj_mat: [b, v, v, e]
        :param node_attr: [b, v, h]
        """
        adj_mats, node_attr = graph.get_adj_mat_and_node_features(self.hidden_layer_size)

        node_features = self.ggnn(node_attr, adj_mats) # [b,v, h2]

        mean_log_var = self.ggnn_top(node_features)  # [b,h1]
        return mean_log_var


class Decoder(base_parameterised_distribution.BaseParameterisedDistribution):
    def __init__(self, max_num_nodes: int, latent_space_dim: int, run_graph_matching_flag: bool):
        super().__init__()


        *_, final_hidden_dim = graph_datastructure.BaseMolecularGraphs.get_data_locations_in_packed_form(max_num_nodes)
        self.parameterizing_net = nn.Sequential(nn.Linear(latent_space_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                                nn.Linear(128,256),nn.BatchNorm1d(256), nn.ReLU(),
                                                nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                                nn.Linear(512, final_hidden_dim)
                                                )
        self.max_num_nodes = max_num_nodes
        self._tilde_structure: typing.Optional[graph_datastructure.LogitMolecularGraphs] = None
        self.run_graph_matching_flag = run_graph_matching_flag

        self._logger: typing.Optional[logging_tools.LogHelper] = None

    def mode(self):
        return self._tilde_structure.calc_distributions_mode()

    def nlog_like_of_obs(self, obs: graph_datastructure.OneHotMolecularGraphs) -> torch.Tensor:
        if self.run_graph_matching_flag:
            this_graph_matched = self._tilde_structure.return_matched_version_to_other(obs)
        else:
            # this version is called NoGM in table 1 of paper.
            this_graph_matched = self._tilde_structure
        nll = this_graph_matched.neg_log_like(obs)

        if self._logger is not None:
            # NB this takes a bit of time but indicates the gain from using the graph matching algorithm.

            if self.run_graph_matching_flag:
                matched_nll = nll
                unmatched_nll = self._tilde_structure.neg_log_like(obs)
            else:
                matched_nll = self._tilde_structure.return_matched_version_to_other(obs).neg_log_like(obs)
                unmatched_nll = nll

            self._logger.add_statistics({
                'sum-nll_matched': matched_nll.sum().item(),
                'sum-nll_unmatched': unmatched_nll.sum().item(),
            })
        return nll

    def update(self, latent_z: torch.Tensor):
        graph_logits_packed = self.parameterizing_net(latent_z)
        self._tilde_structure = graph_datastructure.LogitMolecularGraphs.create_from_nn_prediction(graph_logits_packed,
                                                                                                   self.max_num_nodes)


def make_gvae(latent_space_dim: int, max_num_nodes, cuda_details: utils.CudaDetails, run_graph_matching_flag: bool):
    # Encoder
    encoder = nn_paramterised_dists.NNParamterisedDistribution(EncoderNet(64, cuda_details, T=2, out_dim=2 * latent_space_dim),
                                                    shallow_distributions.IndependentGaussianDistribution())

    # Decoder
    decoder = Decoder(max_num_nodes, latent_space_dim, run_graph_matching_flag)

    # Latent Prior
    latent_prior = shallow_distributions.IndependentGaussianDistribution(torch.zeros(1, 2 * latent_space_dim))

    # VAE
    model = variational.VAE(encoder=encoder, decoder=decoder, latent_prior=latent_prior)
    return model
