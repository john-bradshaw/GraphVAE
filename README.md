# GraphVAE

My implementation of [1].
This implementation is based  on the one at https://github.com/JiaxuanYou/graph-generation [2], but with a sole
focus on molecular graphs.
Note that we do not completely match the hyperparameters/architectures suggested in [1].

1. Simonovsky M and Komodakis N (2018)
 GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders. 
 arXiv [cs.LG]. Available at: http://arxiv.org/abs/1802.03480.
2. Jiaxuan You*, Rex Ying*, Xiang Ren, William L. Hamilton, Jure Leskovec, 
GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Model (ICML 2018)
 
## Notation
`b` the batch size  
`e` the number of edge types  
`v` the number of nodes for adjacency matrix in one graph  
`v*` the number of stacked active nodes in all the graphs    
`E*` the number of edges in all the graphs  
`h` the dimension of the node representation      
`[...]` corresponding array/tensor shape. eg `[2,4]` signifies a 2 by 4 matrix
`g` the number of groups


# TODOS:
[] update GNN submodule reference  
[] debug the vae model: especially check the graph matching part  
[] Add tensorboard hooks to training  
[] Write the sample script. 