#!/usr/bin/env bash


cd scripts
PYTHONPATH=/graph_vae/:/graph_vae/submodules/autoencoders/:/graph_vae/submodules/GNN/ /opt/conda/envs/py36/bin/python train_graphvae.py qm9