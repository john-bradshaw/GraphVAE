
import pytest
from numpy.distutils.system_info import numarray_info

from rdkit import Chem
import numpy as np
import torch


from graph_vae import graph_datastructure

SMILES_LIST = [
    "N#CC1NC11C2CC1C2",
    "OC1=NON=C1OC=O",
    "CC12CCC1CN1CC21",
    "CN=C1OC(=O)CC1N",
    "CC(CO)C(CO)C#N",
    "CC1(C)CC(=O)OC1=N",
    "CC1NC2(CNC12)C#N",
    "OCCOC(=O)N1CC1",
    "O=CC1=CN2CCC2=N1",
    "NC1(COC1=N)C1CN1",
    "CCN1C=NC(=C1)C#C",
    "O=C1CC2COCC1O2",
]


SMILES_LIST_LARGER = [
    "N#CC1NC11C2CC1C2",
    "OC1=NON=C1OC=O",
    "CC12CCC1CN1CC21",
    "CN=C1OC(=O)CC1N",
    "CC(CO)C(CO)C#N",
    "CC1(C)CC(=O)OC1=N",
    "CC1NC2(CNC12)C#N",
    "OCCOC(=O)N1CC1",
    "O=CC1=CN2CCC2=N1",
    "NC1(COC1=N)C1CN1",
    "CCN1C=NC(=C1)C#C",
    "O=C1CC2COCC1O2",
    "CC(=O)CC1(C)COC1",
    "O=CC1=NCCO1",
    "N#CC1(CC2CC2)CO1",
    "CC1=CCCNC1C#N",
    "O=C1NC2CCC3C2N13",
    "N#CC12OC3COC1C23",
    "CNC1=C(N)N=NN1C",
    "OC1(CC=O)CC=CC1",
    "NC1=NC2CC2NC1=O",
    "CN1C(=N)ON=C1C=O",
    "OC1=CC=C(F)C=C1",
    "CC(CO)N1N=NN=N1",
    "C1C2NC2C11CCCO1",
    "CC1(OCCO1)C1CO1",
    "CC#CCC1(C)CN1",
    "O=C1C2CC3CC1C23",
    "C1OC2=NC3C4OC13C24",
    "O=C1NCC11C2CC1O2",
    "CN=C1N=NOC=C1F",
    "CC12OC3C4C3C1(C)C24",
    "CC1=NC=C(C=C1)C#N",
    "C1CC11CC2N=COC12",
    "C1OC23CC(C=C2)C13",
    "O=CNC(C#N)C1CC1",
    "CC1(CC(O)C#C)CC1",
    "CCCCCN=COC",
    "CC1C2C3CC2(C#N)C13",
    "O=C1OC23CC(C2)C13",
    "CC1CCC(C=O)=C1N",
    "CC1=NC(=CC=C1)C#N",
    "COC1=C(OC=N1)C#N",
    "OC1C(COC1=O)C#N",
    "CC(C#C)C1COCO1",
    "OC1C2C(C1O)N2C=O",
    "CCCOC1C2CCC12",
    "CC(C1CCC1)C(C)=O",
    "OC12C3CC1C(=O)C2C3",
    "CC1=NC(=N)C(N)=NN1",
]


def canonicalize(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)


def randomly_permute(ds: graph_datastructure.BaseMolecularGraphs, max_num_nodes: int, rng: np.random.RandomState):
    all_perms = []
    rng = np.random.RandomState(56)
    for _ in range(ds.num_graphs):
        perm = np.zeros((max_num_nodes, max_num_nodes), dtype=np.float32)
        perm[np.arange(max_num_nodes), rng.permutation(max_num_nodes)] = 1.
        all_perms.append(perm)
    all_perms = np.stack(all_perms)
    all_perms = torch.tensor(all_perms)
    ds = ds.return_permuted(all_perms)
    return ds, all_perms


@torch.no_grad()
def mpm_test(smi_list, adj_noise, node_attr_noise, edge_atr_noise, rng: np.random.RandomState, max_node_size=15):
    """
    This tests the max pooled matching algorithm.
    The principle behind this test is from Table 2 of the GraphVAE
    paper (open review) version.

    1. We Load in SMILES_LIST as OH graph.
    2. Randomly permute the points.
    3. Add Gaussian noise to the relevant tensors and re-normalize/truncate.
    4. Run matching to permutation stage -- but due to equivalent nodes being swapped do this via checking
     individual components rather than checking permutation found is its own inverse.
    we applied!
    """
    torch.manual_seed(rng.choice(10000))

    # 1.
    ds = graph_datastructure.OneHotMolecularGraphs.create_from_smiles_list(smi_list, padding_size=max_node_size)

    # 2.
    permuted_ds, permutations = randomly_permute(ds, max_node_size, np.random.RandomState(rng.choice(10000)))

    # 3.
    # adjacency matrices
    # ADd noise
    # Clip, and re-normalize
    # reset diagonals when important.
    assert {adj_noise, node_attr_noise, edge_atr_noise} == {0.}, "Noisy perturbation version not implemented yet"

    # 4.
    permutation_found = permuted_ds._return_matching_permutation(ds.adj_matrices_special_diag, permuted_ds.adj_matrices_special_diag,
                    ds.node_atr_matrices, permuted_ds.node_atr_matrices, ds.edge_atr_tensors, permuted_ds.edge_atr_tensors)
    repermuted_permuted_ds = permuted_ds.return_permuted(permutation_found)
    rp_ds = repermuted_permuted_ds

    # 5. Check matching
    # I'm not actually too sure what the matching accuracies reported in the paper relate to.
    # Reasonably certain they are not looking at complete graph matches as they report decimal points for the percentages
    # but have evaluated only on 100 graphs.
    # Below I look at the mean of all nodes, edges and adjacency matrix
    # elements after converting from one-hot. This means we are assessing the matching of edge attributes that are
    # not necessary used due to the corresponding point in the adjacency matrix being zero...
    bsize = rp_ds.adj_matrices_special_diag.shape[0]

    adj_matching = (rp_ds.adj_matrices_special_diag.contiguous().view(bsize, -1) ==
                    ds.adj_matrices_special_diag.contiguous().view(bsize, -1)).type(torch.float32)
    num_adjs = adj_matching.shape[1]

    node_attr_matching = (rp_ds.node_atr_matrices.contiguous().argmax(dim=-1).contiguous().view(bsize, -1) ==
                          ds.node_atr_matrices.contiguous().argmax(dim=-1).contiguous().view(bsize, -1)).type(torch.float32)
    node_attr_num = node_attr_matching.shape[1]

    # nb contiguous needed becuase argmax on all zero rows will different otherwise.
    edge_matching = (rp_ds.edge_atr_tensors.contiguous().argmax(dim=-1).view(bsize, -1) ==
                     ds.edge_atr_tensors.contiguous().argmax(dim=-1).contiguous().view(bsize, -1)).type(torch.float32)
    edge_matching_num = edge_matching.shape[1]

    proportion_matched = adj_matching.mean() * num_adjs + node_attr_matching.mean() * node_attr_num + edge_matching.mean() * edge_matching_num
    proportion_matched = proportion_matched / (num_adjs + node_attr_num + edge_matching_num)

    return proportion_matched


def test_to_and_from_smi():
    """
    We shall test that we create graph features properly and that we can compute SMILES from graph correctly by just
     checking that we can go back and forth between the two representations.
    """

    MAX_NUM_NODES = 9

    ds = graph_datastructure.OneHotMolecularGraphs.create_from_smiles_list(SMILES_LIST, padding_size=MAX_NUM_NODES)

    # The next step is not strictly necessary but makes sure that still does it after random permuation.
    ds, _ = randomly_permute(ds, MAX_NUM_NODES, np.random.RandomState(56))

    smiles_back = ds.to_smiles_strings()

    orig_smiles_canon = [canonicalize(s) for s in SMILES_LIST]
    smiles_back_canon = [canonicalize(s) for s in smiles_back]
    assert orig_smiles_canon == smiles_back_canon


def test_matching_no_noise():
    """
    permutes the adj matrices etc and makes sure that it can reassemble them using the matched algorithm.
    """
    prop = mpm_test(SMILES_LIST_LARGER, 0., 0., 0., np.random.RandomState(45), max_node_size=15)
    assert prop > 0.9


def test_similarity_function():
    """
    tests my use of einsum! by checking the results against coding out the loop by hand.
    """
    # set seeds
    rng = np.random.RandomState(45)
    torch.manual_seed(rng.choice(10000))
    max_node_size = 15

    # Create graph and premuted graph to assess the similarity function on.
    ds = graph_datastructure.OneHotMolecularGraphs.create_from_smiles_list(SMILES_LIST, padding_size=max_node_size)
    permuted_ds, _ = randomly_permute(ds, max_node_size, np.random.RandomState(rng.choice(10000)))

    adj = ds.adj_matrices_special_diag
    adj_tilde = permuted_ds.adj_matrices_special_diag
    edge_attr = ds.edge_atr_tensors
    edge_attr_tilde = permuted_ds.edge_atr_tensors
    node_attr = ds.node_atr_matrices
    node_attr_tilde  = permuted_ds.node_atr_matrices

    # Get the calculated s_tensor
    print("Doing it via einsum")
    s_tensor_calculated = permuted_ds._calc_similarity_term_between_two_graphs(adj, adj_tilde, node_attr, node_attr_tilde,
                                                                    edge_attr, edge_attr_tilde).numpy()

    # Now do the same thing via numpy and a loop
    print("Doing it via for loop")
    adj = adj.numpy()
    adj_tilde = adj_tilde.numpy()
    edge_attr = edge_attr.numpy()
    edge_attr_tilde = edge_attr_tilde.numpy()
    node_attr = node_attr.numpy()
    node_attr_tilde = node_attr_tilde.numpy()

    batchsize, num_nodes, num_nodes_p2 = adj.shape
    assert num_nodes == num_nodes_p2
    assert num_nodes == adj_tilde.shape[1]
    assert num_nodes == adj_tilde.shape[2]
    s_tensor_via_loop = np.zeros((batchsize, num_nodes, num_nodes, num_nodes, num_nodes), dtype=np.float32)

    for n in range(batchsize):
        for i in range(num_nodes):
            for j in range(num_nodes):
                for a in range(num_nodes):
                    for b in range(num_nodes):
                        term = 0.
                        if (i != j) and (a!= b):
                            term += ((edge_attr[n, i,j, :] @ edge_attr_tilde[n,a,b]) * adj[n, i, j] * adj_tilde[n,a,b]
                                * adj_tilde[n,a,a] * adj_tilde[n,b,b])
                        if (i == j) and (a == b):
                            term += ((node_attr[n,i,:] @ node_attr_tilde[n,a,:]) * adj_tilde[n,a,a])

                        s_tensor_via_loop[n, i, j, a, b] = term

    print("checking equal...")
    np.testing.assert_array_equal(s_tensor_calculated, s_tensor_via_loop)


def test_permutation():
    """tests permutation by permuting and then permuting by transpose to make sure back to original."""
    # Setup and Permute once
    max_num_nodes = 9
    rng = np.random.RandomState(100)
    ds = graph_datastructure.OneHotMolecularGraphs.create_from_smiles_list(SMILES_LIST, padding_size=max_num_nodes)
    permuted_ds, permutations = randomly_permute(ds, max_num_nodes, np.random.RandomState(rng.choice(10000)))

    # Now permute back
    permutations_t = permutations.permute(0,2,1)
    rp_ds = permuted_ds.return_permuted(permutations_t)

    # Now check match
    bsize = rp_ds.adj_matrices_special_diag.shape[0]
    adj_matching = (rp_ds.adj_matrices_special_diag.contiguous().view(bsize, -1) ==
                    ds.adj_matrices_special_diag.contiguous().view(bsize, -1)).all(dim=-1)
    node_attr_matching = (rp_ds.node_atr_matrices.contiguous().view(bsize, -1) ==
                          ds.node_atr_matrices.contiguous().view(bsize, -1)).all(dim=-1)
    edge_matching = (rp_ds.edge_atr_tensors.contiguous().view(bsize, -1) ==
                     ds.edge_atr_tensors.contiguous().view(bsize, -1)).all(dim=-1)
    matches = adj_matching & node_attr_matching & edge_matching

    proportion_matched = torch.mean(matches, dtype=torch.float32).item()
    assert proportion_matched == 1.


def test_mpm():
    """
    tests that I wrote the array version of mpm correctly by assessing against a slow multiple for loop version written
    in numpy.
    """
    num_graphs = 5
    num_nodes = 15
    ds = graph_datastructure.LogitMolecularGraphs(torch.ones(num_graphs, num_nodes, num_nodes),
                                                  torch.ones(num_graphs, num_nodes, num_nodes, graph_datastructure.CHEM_DETAILS.num_bond_types),
                                                  torch.ones(num_graphs, num_nodes, graph_datastructure.CHEM_DETAILS.num_node_types))
    # ^ dummy graph with no information in it.

    s_tensor = np.load('s_tensor_for_testing.npy')[:num_graphs]  # one I prepared earlier! This one only has ones/zeros in it.
    # Comes from SMILES LARGER LIST.
    num_iter = ds.mpm_iterations
    batch_size, num_nodes, nn2, nn3, nn4 = s_tensor.shape
    assert {num_nodes, nn2, nn3, nn4} == {num_nodes}, "all should be 9 number of  nodes."

    print("Evaluating via loop (this is slow...)")
    x_arr_from_np_loop = 1./num_nodes * np.ones((batch_size, num_nodes, num_nodes))
    for n in range(batch_size):

        x_sub = x_arr_from_np_loop[n]
        for iter in range(num_iter):

            x_temp = np.empty_like(x_sub)
            # Over indices.
            for i in range(num_nodes):
                for a in range(num_nodes):

                    sum_ = 0.
                    for j in range(num_nodes):
                        if j == i:
                            continue
                        sum_ += np.max([x_sub[j,b]*s_tensor[n,i,j,a,b] for b in range(num_nodes) if b != a])

                    x_temp[i, a] = x_sub[i, a] * s_tensor[n, i, i, a, a] + sum_

            x_temp = x_temp/np.linalg.norm(x_temp, ord='fro')
            x_sub = x_temp

        x_arr_from_np_loop[n] = x_sub

    print("Evaluating via class")
    x_arr_from_ds = ds._run_mpm(torch.from_numpy(s_tensor)).numpy()

    print("Done!")
    np.testing.assert_array_almost_equal(x_arr_from_np_loop, x_arr_from_ds)

