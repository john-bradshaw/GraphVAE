
import datetime
from contextlib import contextmanager


import numpy as np
import tqdm
import torch
from torch import optim
from torch.utils import data

import tensorboardX

from graph_neural_networks.core import utils
import autoencoders as ae

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


class TbLoggerWrapper:
    """
    Wraps the TB logger and a step counter so that the step can be automatically inserted when logging to Tensorboard.
    """
    def __init__(self, name):
        self.logger = tensorboardX.SummaryWriter(name)
        self.iter = [0]
        self._epoch_storage = {}

    def close(self):
        self.logger.close()

    def add_step(self):
        self.iter[0] += 1

    def add_scalar(self, name, value):
        self.logger.add_scalar(name, value, self.iter[0])

    def add_image(self, tag, img_tensor, dataformats='CHW'):
        self.logger.add_image( tag, img_tensor, global_step= self.iter[0],dataformats=dataformats)

    def add_all_key_vals(self, dict_of_vals_to_add):
        for k, v in dict_of_vals_to_add.items():
            self.add_scalar(k, v)

    def epoch_counting_clr(self):
        self._epoch_storage = {}

    def epoch_counting_add_values(self, dict_of_vals_to_add):
        for k,v in dict_of_vals_to_add:
            self._epoch_storage[k] = v + self._epoch_storage.get(k, default=0)

    def epoch_counting_average_for_epoch(self):
        batchsizes = self._epoch_storage.pop('batchsize')
        for k,v in self._epoch_storage:
            self.add_scalar(k, v/float(batchsizes))


def train_one_epoch(vae, optimizer, dataloader, torch_device, add_step_func):
    with tqdm.tqdm(dataloader, total=len(dataloader)) as t:
        for data in t:
            data.to_inplace(torch_device)
            optimizer.zero_grad()
            elbo = vae.elbo(data).mean()
            loss = -elbo
            loss.backward()
            optimizer.step()
            add_step_func()
            t.set_postfix(elbo=elbo.item())  # update the progress bar


@torch.no_grad()
def vae_val(vae,  dataloader, torch_device):
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


@torch.no_grad()
def sample_graphs_from_prior(vae, tb_logger, latent_space_dim):
    """
    this function samples ten graphs from prior and plots them in Tensorboard as the training progresses.
    """
    from rdkit.Chem import Draw

    # Sample a batch of new graphs at once.
    z_sample = torch.randn(10, latent_space_dim)
    vae.decoder.update(z_sample)
    m: gv.OneHotMolecularGraphs = vae.decoder.mode()

    mols = m.to_molecules()
    for i, mol in enumerate(mols):
        try:
            img_canvas = np.array(Draw.MolToImage(mol, size=(400, 400), fitImage=True))
        except:
            img_canvas = np.zeros((400, 400, 3))
        tb_logger.add_image(f"sample_{i}", img_canvas, dataformats='HWC')


def main(params: Params):
    # Set up log writers
    time_of_run = f"_{datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
    train_writer = TbLoggerWrapper("tb_logs/train" + time_of_run)
    train_log_helper = ae.LogHelper([train_writer.add_all_key_vals])
    val_writer = TbLoggerWrapper("tb_logs/val" + time_of_run)
    val_writer.iter = train_writer.iter
    val_log_helper = ae.LogHelper([val_writer.epoch_counting_add_values])


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
    vae = gv.make_gvae(params.latent_space_dim, params.max_num_nodes, cuda_details, run_graph_matching_flag=True)
    vae = cuda_details.return_cudafied(vae)
    optimizer = optim.Adam(vae.parameters(), lr=params.adam_lr)

    def setup_for_train():
        vae.train()
        vae._logger_manager = train_log_helper

    @contextmanager
    def validation_mode():
        # this context sets up a tensorboard logger to collect values as we iterate through the batches but will only
        # o/p the final computed batch at the end to the tensorboard.
        val_writer.epoch_counting_clr()
        vae._logger_manager = val_log_helper
        yield
        val_writer.epoch_counting_average_for_epoch()

    # Train!
    for epoch_num in range(params.num_epochs):
        print(f"Beginning epoch {epoch_num}")
        setup_for_train()
        train_one_epoch(vae, optimizer, train_dataloader, cuda_details.device_str, add_step_func=train_writer.add_step)

        vae.eval()
        with validation_mode:
            vae_val(vae, valid_dataloader, cuda_details.device_str)
        sample_graphs_from_prior(vae, val_writer, params.latent_space_dim)


    torch.save({
        "vae": vae.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, "graph_vae_weights" + time_of_run + ".pth.pick")
    train_writer.close()
    val_writer.close()




if __name__ == '__main__':
    main(Params())
