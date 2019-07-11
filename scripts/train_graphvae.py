
import numpy as np
import tqdm
import torch
from torch import optim
from torch.utils import data

from graph_neural_networks.core import utils
import graph_vae as gv


class Params:
    def __init__(self):
        self.rng = np.random.RandomState(42)

        # Model Details
        self.latent_space_dim = 40
        self.max_num_nodes = 9

        # Training details
        self.num_epochs = 25
        self.batch_size = 32
        self.adam_lr = 1e-3


def train_one_epoch(vae, optimizer, dataloader, torch_device):
    vae.train()
    with tqdm.tqdm(dataloader, total=len(dataloader)) as t:
        for data in t:
            data.to_inplace(torch_device)
            optimizer.zero_grad()
            elbo = vae.elbo(data).mean()
            loss = -elbo
            loss.backward()
            optimizer.step()
            t.set_postfix(elbo=elbo.item())  # update the progress bar

@torch.no_grad()
def vae_val(vae,  dataloader, torch_device):
    vae.eval()
    num_seen = 0.
    running_total = 0.
    for data in tqdm.tqdm(dataloader, desc="validation"):
        data.to_inplace(torch_device)
        elbo = vae.elbo(data)

        running_total += elbo.sum().item()
        num_seen += elbo.shape[0]

    avg_elbo = float(running_total) / num_seen
    print(f"The average ELBO over the validation dataset is {avg_elbo}")
    return  avg_elbo


def main(params: Params):
    # Get Dataset and break down into train and validation datasets
    dataset = gv.SmilesDataset("../qm9_smiles.txt")
    permutation = params.rng.permutation(len(dataset))
    train_dataset, valid_dataset, test_dataset = dataset.split_dataset(permutation[:-20000], permutation[-20000:-10000],
                                                                       permutation[-10000:])

    # Get Dataloaders
    num_workers = 0  # we are going to convert from SMILES to graphs on the fly hence useful if we do this over several
    # processes.
    collate_func = lambda list_of_smiles: gv.OneHotMolecularGraphs.create_from_smiles_list(list_of_smiles,
                                                                                           params.max_num_nodes)
    train_dataloader = data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=num_workers,
                                       collate_fn=collate_func)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_func)

    # Create model and optimizer
    cuda_details = utils.CudaDetails(use_cuda=torch.cuda.is_available())
    vae = gv.make_gvae(params.latent_space_dim, params.max_num_nodes, cuda_details)
    vae = cuda_details.return_cudafied(vae)
    optimizer = optim.Adam(vae.parameters(), lr=params.adam_lr)

    # Train!
    for epoch_num in range(params.num_epochs):
        print(f"Beginning epoch {epoch_num}")
        train_one_epoch(vae, optimizer, train_dataloader, cuda_details.device_str)
        vae_val(vae, valid_dataloader, cuda_details.device_str)

    torch.save({
        "vae": vae.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, "graph_vae_weights.pth.pick")


if __name__ == '__main__':
    main(Params())
