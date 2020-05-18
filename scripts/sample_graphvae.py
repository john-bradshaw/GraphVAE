"""Train GraphVAE

Usage:
  sample_graphvae.py <weights_name> [--num_samples=<num>]


Options:
    --num_samples=<num>  Maximum number of nodes [default: 20000].
"""

from docopt import docopt

import datetime
from os import path

import numpy as np
import tqdm
import torch

from graph_neural_networks.core import utils

import graph_vae as gv


@torch.no_grad()
def main():
    arguments = docopt(__doc__)
    weights_name = arguments['<weights_name>']
    num_samples = int(arguments['--num_samples'])
    time_of_run = f"_{datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
    cuda_details = utils.CudaDetails(use_cuda=torch.cuda.is_available())

    print(f"Will load checkpoint {weights_name} and generate {num_samples} samples."
          f"Will be saved under {time_of_run}")

    # Load the weights
    saved_state = torch.load(weights_name, map_location=cuda_details.device_str)

    # Set datastuctures
    dataset_name = saved_state['params']['dataset_name']
    print(f"Dataset name: {dataset_name}")
    gv.CHEM_DETAILS.set_for_dataset(saved_state['params']['dataset_name'])

    # Create the network
    print(f"Weights came with the following params: {saved_state['params']}")
    num_ggnn_steps = 3 if dataset_name in {'zinc', "zinc-20"} else 2
    ggnn_hidden_size = 256 if dataset_name in {'zinc', "zinc-20"} else 64
    vae = gv.make_gvae(saved_state['params']['latent_space_dim'],
                       saved_state['params']['max_num_nodes'], cuda_details,
                       run_graph_matching_flag=saved_state['params']['run_graph_matching'], T=num_ggnn_steps,
                       graph_hidden_layer_size=ggnn_hidden_size)
    vae = cuda_details.return_cudafied(vae)
    vae.load_state_dict(saved_state['vae'])

    # Now go through and sample!
    sample_batch_size = 100
    num_batches = int(np.ceil(num_samples/sample_batch_size))
    sampled_smiles = []
    for _ in tqdm.tqdm(range(num_batches), desc="Sampling SMILES"):
        z_sample = torch.randn(sample_batch_size, saved_state['params']['latent_space_dim']).to(cuda_details.device_str)
        vae.decoder.update(z_sample)
        m: gv.OneHotMolecularGraphs = vae.decoder.mode()
        smi_strs = m.to_smiles_strings(add_implicit_hydrogen=False)
        sampled_smiles += smi_strs

    # Now write out!
    sampled_smiles = sampled_smiles[:num_samples]
    save_name = f"SMILES_SAMPLES_{path.basename(weights_name).split('.')[0]}_{time_of_run}.txt"
    with open(save_name, 'w') as fo:
        fo.writelines('\n'.join(sampled_smiles))


if __name__ == '__main__':
    main()
