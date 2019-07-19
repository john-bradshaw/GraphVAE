"""Evaluate Reconstructions

Usage:
  eval_reconstruction.py <weight-path> --sample_z


Options:
    --sample_z  If set then sample z when reconstructing.

"""


import datetime
import json

from docopt import docopt
import tqdm
import torch
from torch.utils.data import DataLoader

from graph_neural_networks.core import utils

import graph_vae as gv
import train_graphvae


def main(params: train_graphvae.Params, weights_path: str, sample_z_flag: bool):
    # Get Dataset and break down into train and validation datasets
    dataset = gv.SmilesDataset("../qm9_smiles.txt")
    permutation = params.rng.permutation(len(dataset))
    train_dataset, valid_dataset, test_dataset = dataset.split_dataset(permutation[:-20000], permutation[-20000:-18000],
                                                                       permutation[-10000:])

    # Get Dataloaders
    num_workers = 0  # we are going to convert from SMILES to graphs on the fly hence useful if we do this over several
    # processes.
    collate_func = lambda list_of_smiles: gv.OneHotMolecularGraphs.create_from_smiles_list(list_of_smiles,
                                                                                           params.max_num_nodes)
    valid_dataloader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False,
                                       num_workers=num_workers,
                                       collate_fn=collate_func)

    # Create model and optimizer
    cuda_details = utils.CudaDetails(use_cuda=torch.cuda.is_available())
    vae = gv.make_gvae(params.latent_space_dim, params.max_num_nodes, cuda_details,
                       run_graph_matching_flag=params.run_graph_matching)
    vae = cuda_details.return_cudafied(vae)

    # Load in weights
    weights = torch.load(weights_path, map_location=cuda_details.device_str)
    vae.load_state_dict(weights['vae'])

    # Go through and see about reconstructing
    reconstructed_smiles = []
    for data in tqdm.tqdm(valid_dataloader, desc="validation"):
        data.to_inplace(cuda_details.device_str)
        reconstruction: gv.OneHotMolecularGraphs = vae.reconstruct_no_grad(data, sample_z=sample_z_flag, sample_x=False)
        # ^ set sample_x to False such that you take the mode from the decoder.
        smiles = reconstruction.to_smiles_strings(add_implicit_hydrogen=False)
        reconstructed_smiles += smiles

    # Now go and construct a list showing original and reconstructed.
    valid_smiles = valid_dataset.data
    orig_recon = list(zip(valid_smiles, reconstructed_smiles))
    save_name = f"reconstructed_results_{datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.json"
    with open(save_name, 'w') as fo:
        json.dump(orig_recon, fo)

    print(f"Run, saving reconstruction results at {save_name}.")


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print(f"Running with arguments: {arguments}")
    main(train_graphvae.Params(), arguments['weight-path'], arguments['sample_z'])
