"""Train GraphVAE

Usage:
  train_graphvae.py <dataset_name> [--mpm] [--max_num_nodes=<num>]


Options:
    --mpm   run the MPM graph matching routine.
    --max_num_nodes=<num>  Maximum number of nodes [default: 9].
"""
from docopt import docopt

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
        arguments = docopt(__doc__)
        self.dataset_name = arguments['<dataset_name>']

        self.rng = np.random.RandomState(3743)
        torch.manual_seed(self.rng.choice(10000))

        # Model Details
        self.run_graph_matching = arguments['--mpm']
        self.latent_space_dim = 40
        self.max_num_nodes = int(arguments['--max_num_nodes'])
        self.beta = 1. / 40.
        # in their paper they had beta=1. (apart from appendix). However, I'm not too sure whether they were also
        # averaging KL as this was done eg in the GrammarVAE paper. Hence here set to that effect.

        # Training details
        self.num_epochs = 25
        self.batch_size = 32
        self.adam_lr = 1e-3
        self.adam_beta1 = 0.5

        print(f"Dataset name is {self.dataset_name} and we are running graph matching is {self.run_graph_matching}")
        print(f"Max num nodes is {self.max_num_nodes}")

    def save_weights_name(self, time_of_run, epoch_num=None):
        mid_ = "_with_graph_matching" if self.run_graph_matching else ""
        epoch_str = "" if epoch_num is None else f"_epoch{epoch_num}"
        return "weights/gvae" + mid_ + epoch_str + time_of_run + f"_{self.dataset_name}.pth.pick"

    def get_params_to_save(self):
        return {k:v for k, v in self.__dict__.items() if isinstance(v, (int, float, str, bool))}


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

    def add_all_key_vals_as_scalars(self, dict_of_vals_to_add):
        batchsize = dict_of_vals_to_add.get('raw-batchsize')
        for k, v in dict_of_vals_to_add.items():
            type_of_stat, name = k.split('-')
            if type_of_stat == 'sum':
                self.add_scalar(name, v/batchsize)
            else:
                self.add_scalar(name, v)

    def epoch_counting_clr(self):
        self._epoch_storage = {}

    def epoch_counting_add_values(self, dict_of_vals_to_add):
        for k,v in dict_of_vals_to_add.items():
            type_of_stat, name = k.split('-')
            additive_term = v
            self._epoch_storage[name] = additive_term + self._epoch_storage.get(k, 0)

    def epoch_counting_average_for_epoch(self):
        batchsizes = self._epoch_storage.pop('batchsize')
        for k,v in self._epoch_storage.items():
            self.add_scalar(k, v/float(batchsizes))


def train_one_epoch(vae, optimizer, dataloader, torch_device, add_step_func, beta):
    with tqdm.tqdm(dataloader, total=len(dataloader)) as t:
        for data in t:
            data.to_inplace(torch_device)
            optimizer.zero_grad()
            elbo = vae.elbo(data, beta=beta).mean()
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
def sample_graphs_from_prior(vae, tb_logger, latent_space_dim, torch_device):
    """
    this function samples num_samples graphs from prior and plots them in Tensorboard as the training progresses.
    """
    from rdkit.Chem import Draw
    from rdkit import Chem

    # Sample a batch of new graphs at once.
    num_samples = 100
    num_to_draw = 30
    z_sample = torch.randn(num_samples, latent_space_dim).to(torch_device)
    vae.decoder.update(z_sample)
    m: gv.OneHotMolecularGraphs = vae.decoder.mode()

    # We will draw out the molecules into Tensorboard's image section.
    mols = m.to_molecules()
    for i, mol in enumerate(mols[:num_to_draw]):
        try:
            img_canvas = np.array(Draw.MolToImage(mol, size=(400, 400), fitImage=True))
        except:
            img_canvas = np.zeros((400, 400, 3))
        tb_logger.add_image(f"sample_{i}", img_canvas, dataformats='HWC')

    # We'll measure and print out how many of these can be parsed by RDKIT
    success = 0
    for mol in mols:
        try:
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))
            # ^nb can write it out the first time but not the second.
            success += 1
        except Exception:
            pass
    print(f"Sampled {num_samples} graphs from prior, of these {success} were able to"
          f" be written to SMILES (proportion of {success/num_samples:.3f})")



@torch.no_grad()
def check_reconstructions(vae, tb_logger, smiles, torch_device, max_num_nodes):
    from rdkit.Chem import Draw
    from rdkit import Chem

    # Sample a batch of new graphs at once.
    data = gv.OneHotMolecularGraphs.create_from_smiles_list(smiles, max_num_nodes)
    data.to_inplace(torch_device)
    m: gv.OneHotMolecularGraphs = vae.reconstruct_no_grad(data, sample_z=False, sample_x=False)
    mols = m.to_molecules()
    for i, (orig_smi, mol) in enumerate(zip(smiles, mols)):
        try:
            orig_mol = Chem.MolFromSmiles(orig_smi)
            img_canvas = np.array(Draw.MolToImage(mol, size=(400, 400), fitImage=True))
            img_canvas_orig = np.array(Draw.MolToImage(orig_mol, size=(400, 400), fitImage=True))
            img_canvas = np.concatenate([img_canvas_orig, img_canvas], axis=1)
        except:
            img_canvas = np.zeros((400, 800, 3))
        tb_logger.add_image(f"reconstruction_{i}", img_canvas, dataformats='HWC')


def main(params: Params):
    # Set up log writers
    time_of_run = f"_{datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
    print(f"Running at: {time_of_run}")
    train_writer = TbLoggerWrapper("tb_logs/train" + time_of_run)
    train_log_helper = ae.LogHelper([train_writer.add_all_key_vals_as_scalars])
    val_writer = TbLoggerWrapper("tb_logs/val" + time_of_run)
    val_writer.iter = train_writer.iter
    val_log_helper = ae.LogHelper([val_writer.epoch_counting_add_values])

    # Get Dataset and break down into train and validation datasets
    train_dataset, valid_dataset, test_dataset = gv.get_dataset(params.dataset_name)

    # Set details about atoms and elements
    gv.CHEM_DETAILS.set_for_dataset(params.dataset_name)

    # Get Dataloaders
    num_workers = 3
    # ^ we are going to convert from SMILES to graphs on the fly hence useful if we do this over several processes.
    collate_func = lambda list_of_smiles: gv.OneHotMolecularGraphs.create_from_smiles_list(list_of_smiles,
                                                                                           params.max_num_nodes)
    train_dataloader = data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=num_workers,
                                       collate_fn=collate_func)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_func)

    # Create model and optimizer
    cuda_details = utils.CudaDetails(use_cuda=torch.cuda.is_available())
    num_ggnn_layers = 3 if params.dataset_name in {'zinc', "zinc-20"} else 2
    ggnn_hidden_size = 256 if params.dataset_name in {'zinc', "zinc-20"} else 64
    vae = gv.make_gvae(params.latent_space_dim, params.max_num_nodes, cuda_details,
                       run_graph_matching_flag=params.run_graph_matching, T=num_ggnn_layers, graph_hidden_layer_size=ggnn_hidden_size)
    vae = cuda_details.return_cudafied(vae)
    optimizer = optim.Adam(vae.parameters(), lr=params.adam_lr, betas=(params.adam_beta1, 0.999))

    def setup_for_train():
        vae.train()
        vae._logger_manager = train_log_helper
        vae.decoder._logger = None  # < to save time we dont want to compute forward and backwards.

    @contextmanager
    def validation_mode():
        # this context sets up a tensorboard logger to collect values as we iterate through the batches but will only
        # o/p the final computed batch at the end to the tensorboard.
        val_writer.epoch_counting_clr()
        vae._logger_manager = val_log_helper
        #vae.decoder._logger = val_log_helper
        yield
        val_writer.epoch_counting_average_for_epoch()

    # Train!
    for epoch_num in range(params.num_epochs):
        print(f"Beginning epoch {epoch_num}")
        setup_for_train()
        train_one_epoch(vae, optimizer, train_dataloader, cuda_details.device_str, add_step_func=train_writer.add_step,
                        beta=params.beta)

        # After each epoch we look at some validation metrics.
        vae.eval()
        with validation_mode():
            vae_val(vae, valid_dataloader, cuda_details.device_str)
        sample_graphs_from_prior(vae, val_writer, params.latent_space_dim, cuda_details.device_str)
        check_reconstructions(vae, val_writer, valid_dataset.data[:30], cuda_details.device_str, params.max_num_nodes)

        torch.save({
            "vae": vae.state_dict(),
            "optimizer": optimizer.state_dict(),
            'params': params.get_params_to_save()
        }, params.save_weights_name(time_of_run, epoch_num))

    # Save weights and closer.
    torch.save({
        "vae": vae.state_dict(),
        "optimizer": optimizer.state_dict(),
        'params': params.get_params_to_save()
    }, params.save_weights_name(time_of_run))
    train_writer.close()
    val_writer.close()


if __name__ == '__main__':
    main(Params())
