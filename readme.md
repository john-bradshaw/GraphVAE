# GraphVAE

My implementation of [1].
This implementation is based  on the one at https://github.com/JiaxuanYou/graph-generation, but with a sole
focus on molecular graphs.

1. Simonovsky M and Komodakis N (2018)
 GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders. 
 arXiv [cs.LG]. Available at: http://arxiv.org/abs/1802.03480.
 
 
## Notation
`b` the batch size  
`e` the number of edge types  
`v` the number of nodes for adjacency matrix in one graph  
`v*` the number of stacked active nodes in all the graphs    
`E*` the number of edges in all the graphs  
`h` the dimension of the node representation      
`[...]` corresponding array/tensor shape. eg `[2,4]` signifies a 2 by 4 matrix
`g` the number of groups


