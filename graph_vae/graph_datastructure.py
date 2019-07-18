
import typing

import numpy as np
from scipy import optimize
from scipy.sparse import csgraph
import torch
from torch.nn import functional as F

from rdkit import Chem

class ChemicalDetails:
    def __init__(self):
        self.atom_types =  ['C', 'N', 'O', 'S', 'Se', 'Si', 'I', 'F', 'Cl', 'Br']
        self.bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]

        self.ele_to_idx = {k:v for v,k  in enumerate(self.atom_types)}
        self.idx_to_ele = {v:k for k,v in self.ele_to_idx.items()}
        self.edge_to_idx = {k:v for v,k  in enumerate(self.bond_types)}
        self.idx_to_edge = {v:k for k,v  in self.edge_to_idx.items()}

    @property
    def num_bond_types(self):
        return len(self.bond_types)

    @property
    def num_node_types(self):
        return len(self.atom_types)


CHEM_DETAILS = ChemicalDetails()
NP_FLOAT_TYPE = np.float32


class BaseMolecularGraphs:
    """
    Base class for representing graphs.

    We represent graphs (representing molecules -- with all this entails ie undirected discrete edges etc) by:
    - adj_matrices_special_diag:
        Adjacency matrix with diagonal representing whether the node exists or not -- ie should
        be one for existing. [b, v, v]
    - edge_atr_tensors:
        edge features, ie usually is one hot encoding of edge type. [b, v, v, h_e]
    - node_atr_matrices:
        node features, ie usually is the one-hot encoding of node type. [b, v, h_n]


    These can be padded but padding must occur after active indices.
    """


    def __init__(self, adj_matrices_special_diag, edge_atr_tensors, node_atr_matrices):
        self.adj_matrices_special_diag = adj_matrices_special_diag
        self.edge_atr_tensors = edge_atr_tensors
        self.node_atr_matrices = node_atr_matrices

        self.mpm_iterations = 75   # MPM = Max pooling matching -- see section 3.4 of original paper.

    @property
    def device_str(self):
        return str(self.adj_matrices_special_diag.device)

    @property
    def num_nodes(self):
        return self.adj_matrices_special_diag.shape[1]

    @property
    def num_graphs(self):
        return self.adj_matrices_special_diag.shape[0]

    @classmethod
    def get_data_locations_in_packed_form(cls, max_node_size):
        """
        :param max_node_size: number of nodes we expect the packed data to reperesent
        """
        # Work out the indices for which the different data lies
        num_indcs_for_adj_mat = max_node_size * (max_node_size + 1) // 2
        num_indcs_for_edge_atr = max_node_size * (max_node_size - 1) // 2 * CHEM_DETAILS.num_bond_types
        num_indcs_for_node_attribute_matrix = max_node_size * CHEM_DETAILS.num_node_types

        total_indcs = num_indcs_for_adj_mat + num_indcs_for_edge_atr + num_indcs_for_node_attribute_matrix
        return num_indcs_for_adj_mat, num_indcs_for_edge_atr, num_indcs_for_node_attribute_matrix, total_indcs

    def to_inplace(self, device):
        self.adj_matrices_special_diag = self.adj_matrices_special_diag.to(device)
        self.edge_atr_tensors = self.edge_atr_tensors.to(device)
        self.node_atr_matrices = self.node_atr_matrices.to(device)
        return self

    def return_permuted(self, permutation_matrices):
        """
        :param permutation_matrices: [b, v, v]
        """
        batch_size = self.num_graphs
        num_nodes = self.num_nodes

        # ==================
        # Permute the matrices
        new_adj_mat = torch.bmm(permutation_matrices, self.adj_matrices_special_diag)
        permutation_tranpsosed = permutation_matrices.permute(0, 2, 1)
        new_adj_mat = torch.bmm(new_adj_mat, permutation_tranpsosed)

        new_node_atr = torch.bmm(permutation_matrices, self.node_atr_matrices)

        # the edge attribute one is easier to deal with if we first switch the dimensions
        temp_edge_attr = self.edge_atr_tensors.permute(0,3,1,2).contiguous().view(-1, num_nodes, num_nodes)
        new_edge_attr = torch.bmm(permutation_matrices.repeat_interleave(CHEM_DETAILS.num_bond_types, dim=0),temp_edge_attr)
        new_edge_attr = torch.bmm(new_edge_attr, permutation_tranpsosed.repeat_interleave(CHEM_DETAILS.num_bond_types, dim=0))
        new_edge_attr = new_edge_attr.view(batch_size, CHEM_DETAILS.num_bond_types, num_nodes, num_nodes)
        new_edge_attr = new_edge_attr.permute(0,2,3,1)

        return self.__class__(new_adj_mat, new_edge_attr, new_node_atr)

    def _return_matching_permutation(self, adj, adj_tilde, node_attr, node_attr_tilde, edg, edg_tilde):
        """
        Works out a permutation from this one to other and returns this one in that permutations.
        This other should be a fixed graph.
        See section 3.4 of [1] and the appendix A.
        """
        with torch.no_grad():
            s_tensor = self._calc_similairty_term_between_two_graphs(adj, adj_tilde, node_attr,
                                                                     node_attr_tilde, edg, edg_tilde)
            permutation_matrices = self._run_mpm(s_tensor)
        return permutation_matrices

    @torch.no_grad()
    def _run_mpm(self, s_tensor):
        batch_size, num_nodes, *_ = s_tensor.shape
        # ==================
        # Then we run MPM for a set number of iterations
        # init x:
        x = torch.full((batch_size, num_nodes, num_nodes), fill_value=1. / num_nodes, device=str(s_tensor.device),
                       dtype=s_tensor.dtype)  # ^so that it starts with the row and column sums being correct

        for iter in range(self.mpm_iterations):
            for i in range(num_nodes):
                for a in range(num_nodes):
                    summed_max = torch.sum(
                        torch.stack([torch.max(x[:, j, :] * s_tensor[:, i, j, a, :], dim=-1)[0]
                                     for j in range(num_nodes) if j != i]),
                        dim=0)
                    x[:, i, a] = x[:, i, a] * s_tensor[:, i, i, a, a] + summed_max

            x = x / torch.norm(x, dim=(1, 2), keepdim=True)

        # ==================
        # We run the Hungarian algorithm on each member of the batch
        x = x.detach().cpu().numpy()
        cost_matrices = -x
        permutation_matrices = []
        for cost_matrix in cost_matrices:
            row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
            permute_other_to_self_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            permute_other_to_self_matrix[row_ind, col_ind] = 1.
            permutation_matrices.append(permute_other_to_self_matrix)
        permutation_matrices = np.stack(permutation_matrices)
        permutation_matrices = torch.tensor(permutation_matrices).to(str(self.adj_matrices_special_diag.device))
        return permutation_matrices

    @torch.no_grad()
    def _calc_similairty_term_between_two_graphs(self, adj, adj_tilde, node_attr, node_attr_tilde, edg, edg_tilde):
        """
        eqn 4.
        """
        # Get sizes.
        batch_size = edg.shape[0]
        assert batch_size == edg_tilde.shape[0]

        num_nodes = edg.shape[1]
        assert num_nodes == edg_tilde.shape[1]

        # masks
        diag_mat = torch.diag(torch.ones(num_nodes, dtype=torch.uint8, device=str(edg.device)))  # [v, v]
        i_equal_j_mask = diag_mat[None, :, :, None, None].repeat(batch_size, 1, 1, num_nodes, num_nodes)  # [b,v,v,v,v]
        a_equal_b_mask = diag_mat[None, None, None, :, :].repeat(batch_size, num_nodes, num_nodes, 1, 1)  # [b,v,v,v,v]

        # First term.
        term1 = torch.einsum("nijk,nabk,nij,nab,naa,nbb->nijab", edg, edg_tilde, adj, adj_tilde, adj_tilde, adj_tilde)
        # ^ n is batch dimension
        # now zero out according to the indicator function (De Morgan 1 to convert into or):
        term1[i_equal_j_mask] = 0.
        term1[a_equal_b_mask] = 0.

        # Second term
        term2 = torch.einsum("nik,nak,naa->nia", node_attr, node_attr_tilde, adj_tilde)
        term2 = term2[:, :, None, :, None].repeat(1, 1, num_nodes, 1, num_nodes)
        term2[~(i_equal_j_mask & a_equal_b_mask)] = 0.

        s_tensor = term1 + term2
        return s_tensor


class OneHotMolecularGraphs(BaseMolecularGraphs):
    """
    Graphs where we represent the type of node bonds using one hot vector

    The defining tensors represent one-hot encodings of fixed edges and node types, ie a fixed graph.

    Stores data in torch tensors.
    """
    def get_adj_mat_and_node_features(self, min_node_feature_dim):
        """
        The adjacency matrices with one-hot encoding of edge type and the node associated node features.
        """

        node_features = self.node_atr_matrices  # [b,v,h]
        node_feature_dim = node_features.shape[-1]
        if node_feature_dim < min_node_feature_dim:
            node_features = F.pad(node_features, (0, min_node_feature_dim-node_feature_dim), "constant", 0)
        return self.edge_atr_tensors, node_features


    def to_smiles_strings(self, add_implicit_hydrogen=True) -> typing.List[str]:
        """
        Converts graphs to SMILES strings and adds implicit Hydrogens if set.
        If does not parse then returns an empty string.
        """
        out = []
        for m in self.to_molecules():
            try:
                out.append(Chem.MolToSmiles(m, canonical=True, allHsExplicit=add_implicit_hydrogen))
            except Exception as ex:
                out.append("")
        return out

    def to_molecules(self):
        """
        Converts graphs to rdkit Molecules. May not be valid.
        """
        assert self.num_nodes >= 0, "empty graphs"
        assert self.num_graphs >= 0, "no graphs"
        mols = []

        # ==================
        # We'll do one graph at a time
        for b in range(self.num_graphs):

            # ==================
            # Get relevant arrays and convert one hot to label.
            adj = self.adj_matrices_special_diag[b].detach().cpu().numpy()
            node_atr = self.node_atr_matrices[b].detach().cpu().numpy()
            edge_atr = self.edge_atr_tensors[b].detach().cpu().numpy()
            node_labels = np.argmax(node_atr, axis=-1)
            edge_labels = np.argmax(edge_atr, axis=-1)

            # ==================
            # Create molecule and add atoms and bonds
            mol = Chem.RWMol()
            for atom_idx in range(self.num_nodes):
                # If atom does not exist then finished with adding atoms to this molecule
                if adj[atom_idx,atom_idx] == 0:
                    break  # added all atoms that are "ON"

                # Look up correct element type and add an atom of this element to this molecule
                element = CHEM_DETAILS.idx_to_ele[int(node_labels[atom_idx])]
                atm = Chem.Atom(element)
                new_idx = mol.AddAtom(atm)
                assert new_idx == atom_idx, "rdkit not adding atoms indices in order"

                # now go back through all the previous atoms and if a bond exists then add it!
                for atom_prev_indx in range(atom_idx):
                    bond_exists = adj[atom_prev_indx, atom_idx] == 1.  # note how use only top half of adj. matrix
                    if bond_exists:
                        bond_type = CHEM_DETAILS.idx_to_edge[int(edge_labels[atom_prev_indx, atom_idx])]
                        mol.AddBond(atom_idx, atom_prev_indx, bond_type)
            mols.append(mol.GetMol())
        return mols

    @classmethod
    def create_from_smiles_list(cls, smiles_list: typing.List[str], padding_size: int):
        batch_size = len(smiles_list)

        # ==================
        # Create empty tensors
        adj_mat = torch.zeros(batch_size, padding_size, padding_size)
        node_attr = torch.zeros(batch_size, padding_size, CHEM_DETAILS.num_node_types)
        edge_attr = torch.zeros(batch_size, padding_size, padding_size, CHEM_DETAILS.num_bond_types)

        # ==================
        # Go through and convert each SMILES representation to its graph representation
        for batch_idx, mol_smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(mol_smi)
            for atm_idx, atm in enumerate(mol.GetAtoms()):
                this_atms_idx = atm.GetIdx()
                assert atm_idx == this_atms_idx, "rdkit enumerates atoms in non sequential order."

                # diagonal of adjacency matrix
                adj_mat[batch_idx, atm_idx, atm_idx] = 1.

                # The node attributes
                element = atm.GetSymbol()
                node_attr[batch_idx, atm_idx, CHEM_DETAILS.ele_to_idx[element]] = 1.

            # Now the bonds!
            for bnd in mol.GetBonds():
                beg_idx = bnd.GetBeginAtomIdx()
                end_idx = bnd.GetEndAtomIdx()
                adj_mat[batch_idx, beg_idx, end_idx] = 1.
                adj_mat[batch_idx, end_idx, beg_idx] = 1.

                bnd_idx = CHEM_DETAILS.edge_to_idx[bnd.GetBondType()]
                edge_attr[batch_idx, beg_idx, end_idx, bnd_idx] = 1.
                edge_attr[batch_idx, end_idx, beg_idx, bnd_idx] = 1.

        return cls(adj_mat, edge_attr, node_attr)

    def return_matched_version_to_other(self, other):
        with torch.no_grad():
            assert isinstance(other, OneHotMolecularGraphs)
            # ==================
            # First we create S -- we'll do this as a five dimensional tensor (first dim is batch dimension)
            edg = other.edge_atr_tensors
            edg_tilde = self.edge_atr_tensors
            adj = other.adj_matrices_special_diag
            adj_tilde = self.adj_matrices_special_diag
            node_attr = other.node_atr_matrices
            node_attr_tilde = self.node_atr_matrices, dim=-1
        permutation = self._return_matching_permutation(adj, adj_tilde, node_attr, node_attr_tilde, edg, edg_tilde)
        return self.return_permuted(permutation)


class LogitMolecularGraphs(BaseMolecularGraphs):
    """
    A distribution over Graphs.

    The defining tensors represent the logits for the probability of the associated edges,
     ie a distribution over graphs.

    Stores data in Torch Tensors.
    """

    def calc_distributions_mode(self):
        """
        Note:
        * We compress down to remove as much padding as possible.

        * NB does lot of work in numpy
        For equal max probs we just pick the first.
        """
        orig_device = self.device_str

        # ==================
        # Convert to probabilties (from logits)
        adj_probs = F.sigmoid(self.adj_matrices_special_diag).detach().cpu().numpy()
        edge_atr_probs = F.softmax(self.edge_atr_tensors, dim=-1).detach().cpu().numpy()
        node_atr_probs = F.softmax(self.node_atr_matrices, dim=-1).detach().cpu().numpy()

        # ==================
        # We'll do the rest one graph at a time
        batch_size = adj_probs.shape[0]
        graph_sizes = set()
        all_adjs = []
        all_node_atr = []
        all_edge_attr = []
        for i in range(batch_size):
            # ===
            # Calculate which nodes are turned on
            adj = adj_probs[i, :, :]
            indcs_mask = np.diag(adj) >= 0.5
            adj_indcs = np.nonzero(indcs_mask)[0]
            graph_sizes.add(adj_indcs.size)

            # ===
            # Now work out what edges are on.
            # # The maximum spanning tree is introduced in Section 4.1 of [1]
            new_adj = adj[adj_indcs, :][:, adj_indcs]
            np.fill_diagonal(new_adj, 0.)
            mst = csgraph.minimum_spanning_tree(-new_adj).toarray()
            definitely_on_edges = mst < 0
            definitely_on_edges = np.logical_or(definitely_on_edges, definitely_on_edges.T)  # as want it to be symmetric
            other_on_edges = new_adj > 0.5
            adj_matrix = np.logical_or(definitely_on_edges, other_on_edges).astype(NP_FLOAT_TYPE)
            np.fill_diagonal(adj_matrix, 1.)
            all_adjs.append(adj_matrix)

            # ===
            # Now work out the node types for these on nodes
            nodes_of_interest = node_atr_probs[i, adj_indcs]
            nodes_attr = np.zeros_like(nodes_of_interest)
            indcs_on = np.argmax(nodes_of_interest, axis=1)
            nodes_attr[np.arange(indcs_on.shape[0]), indcs_on] = 1.
            all_node_atr.append(nodes_attr)

            # ===
            # Now work out the edge types for the on nodes.
            edges_of_interest = edge_atr_probs[i, adj_indcs, ...][:, adj_indcs, :]
            edge_attr = np.zeros_like(edges_of_interest, dtype=NP_FLOAT_TYPE)
            num_nodes, _, num_edges = edge_attr.shape
            edge_attr = edge_attr.reshape(-1, num_edges)
            edges_on = adj_matrix.flatten()==1
            edge_attr[np.arange(num_nodes*num_nodes)[edges_on], np.argmax(edges_of_interest, axis=-1).flatten()[edges_on]] = 1.
            edge_attr = edge_attr.reshape(num_nodes, num_nodes, num_edges)


            # We now set the bottom of the adjacency matrix to match the top (should be unecessary)
            # --but zeros out diagonal
            edge_attr[np.tril_indices(edge_attr.shape[0])] = 0.
            edge_attr = edge_attr + np.transpose(edge_attr, (1, 0, 2))
            all_edge_attr.append(edge_attr)

        # ==================
        # Go through and pad each graph to correct size.
        num_graphs = len(all_adjs)
        max_size = max(graph_sizes)
        for i in range(num_graphs):
            g_size = all_adjs[i].shape[0]
            all_adjs[i] = np.pad(all_adjs[i], ((0, max_size-g_size), (0, max_size-g_size)), mode='constant',
                                 constant_values=0.)
            all_node_atr[i] = np.pad(all_node_atr[i], ((0, max_size-g_size), (0, 0)), mode='constant',
                                 constant_values=0.)
            all_edge_attr[i] = np.pad(all_edge_attr[i], ((0, max_size-g_size), (0, max_size-g_size), (0, 0)), mode='constant',
                                 constant_values=0.)
        # ==================
        # Stack and tensorfy!
        adj = torch.from_numpy(np.stack(all_adjs)).to(orig_device)
        node_atr = torch.from_numpy(np.stack(all_node_atr)).to(orig_device)
        edge_atr = torch.from_numpy(np.stack(all_edge_attr)).to(orig_device)

        return OneHotMolecularGraphs(adj, edge_atr, node_atr)

    def neg_log_like(self, other: OneHotMolecularGraphs, lambda_a=1., lambda_f=1., lambda_e=1.):
        """
        The negative log likelihood of this distribution over graphs for the instance seen in Other

        Eqn 2 of paper. Use weightings in eqn 3.
        Note that that for consistency with this paper we take the "mean" loss over adjacency and other tensors rather
        than the sum.
        """
        # ==================
        # First we'll look at the Adjacency matrix.
        # Note I *think* (could have understood it wrong)
        # that eqn2 of paper considers both the top and bottom of this matrix.
        # Although the open source implementation https://github.com/snap-stanford/GraphRNN/ seems to not double count.
        # We will double count for below but could change to only count edge probability once.
        batch_size = self.num_graphs
        adj_logits = self.adj_matrices_special_diag.view(-1)  # [bvv]
        adj_truth = other.adj_matrices_special_diag.view(-1)  # [bvv]
        loss1 = F.binary_cross_entropy_with_logits(adj_logits, adj_truth, reduction='none')
        loss1 = torch.sum(loss1.view(batch_size, -1), dim=1) / ((self.num_nodes)**2)

        # ==================
        # We now work out which of the attribute terms we are actually interested in.
        # In paper it says loss is only taken over the matched attribute nodes so do not want to include loss over all.

        # Start with creating the masks for selecting the right parts to take loss over
        with torch.no_grad():
            num_nodes = self.num_nodes
            match_nodes = torch.diagonal(other.adj_matrices_special_diag, dim1=1, dim2=2) == 1.  # [b, v]
            num_nodes_active = torch.sum(match_nodes, dim=1, dtype=torch.float32)  # [b]
            a_norm_one = torch.sum(other.adj_matrices_special_diag, dim=[1,2], dtype=torch.float32)  # b

            node_attr_mask = match_nodes.view(-1)  # [bv]
            node_idx = torch.arange(batch_size, device=self.device_str).repeat_interleave(num_nodes)
            graph_idx_associated_with_considered_node = node_idx[node_attr_mask]

            # edge attribute tensor mask.
            edge_mask = match_nodes[:,:, None] * match_nodes[:, None, :]
            diag_mask = torch.eye(self.num_nodes, self.num_nodes, dtype=torch.uint8, device=self.device_str)[None, :, :].repeat(self.num_graphs, 1, 1)
            edge_mask.masked_fill_(diag_mask, 0)
            edge_mask = edge_mask.view(-1)  # [bvv]
            edge_idx = torch.arange(batch_size, device=self.device_str).repeat_interleave(num_nodes*num_nodes)
            graph_idx_associated_with_considered_edge = edge_idx[edge_mask]

        # Node attribute loss
        node_num_classes = self.node_atr_matrices.shape[-1]
        node_logits = self.node_atr_matrices.contiguous().view(-1, node_num_classes) # [bv, h]
        node_logits = node_logits[node_attr_mask]
        true_node = other.node_atr_matrices.view(-1, node_num_classes)[node_attr_mask]
        loss2 = F.cross_entropy(node_logits, true_node.argmax(dim=1), reduction='none')
        loss2 = (torch.scatter_add(torch.zeros(batch_size, dtype=loss2.dtype, device=loss2.device), 0, graph_idx_associated_with_considered_node, loss2)
                 / num_nodes_active)


        # And edge attribute
        edge_num_classes = self.edge_atr_tensors.shape[-1]
        pred_edge_logits = self.edge_atr_tensors.contiguous().view(-1, edge_num_classes)
        pred_edge_logits = pred_edge_logits[edge_mask]
        true_edge = other.edge_atr_tensors.view(-1, edge_num_classes)[edge_mask]
        loss3 = F.cross_entropy(pred_edge_logits, true_edge.argmax(dim=1), reduction='none')

        loss3 = torch.scatter_add(torch.zeros(batch_size, dtype=loss3.dtype, device=loss3.device), 0, graph_idx_associated_with_considered_edge, loss3)
        loss3 = loss3 / (a_norm_one - num_nodes_active)
        # ^ nb note that we are dividing through by the number of edges that should be on even though we are
        # testing for all the attributes including the edges that are off. This is to match eqn2.

        # ==================
        # Finally sum together!
        loss = lambda_a * loss1 + lambda_f * loss2 + lambda_e * loss3
        return loss

    def return_matched_version_to_other(self, other: OneHotMolecularGraphs):
        with torch.no_grad():
            assert isinstance(other, OneHotMolecularGraphs)
            # ==================
            # First we create S -- we'll do this as a five dimensional tensor (first dim is batch dimension)
            edg = other.edge_atr_tensors
            edg_tilde = F.softmax(self.edge_atr_tensors, dim=-1)
            adj = other.adj_matrices_special_diag
            adj_tilde = F.sigmoid(self.adj_matrices_special_diag)
            node_attr = other.node_atr_matrices
            node_attr_tilde = F.softmax(self.node_atr_matrices, dim=-1)
        permutation = self._return_matching_permutation(adj, adj_tilde, node_attr, node_attr_tilde, edg, edg_tilde)
        return self.return_permuted(permutation)

    @classmethod
    def create_from_nn_prediction(cls, packed_tensor, max_node_size: int):
        """
        :param packed_tensor: [b, h]
        unpack a real valued tensor into predictions for the suitable tensors that predict graph structure
        """
        # Work out where the data lives inside the packed form.
        batch_size = packed_tensor.shape[0]
        (num_indcs_for_adj_mat, num_indcs_for_edge_atr,
         num_indcs_for_node_attribute_matrix, total_indcs) = cls.get_data_locations_in_packed_form(max_node_size)

        # Take the relevant data out of the correct locations
        adj_mat_logits = packed_tensor[:, :num_indcs_for_adj_mat]
        edge_atr_logits = packed_tensor[:, num_indcs_for_adj_mat:num_indcs_for_edge_atr+num_indcs_for_adj_mat]
        node_atr_logits = packed_tensor[:, num_indcs_for_edge_atr+num_indcs_for_adj_mat:]
        assert packed_tensor.shape[1] == total_indcs
        assert node_atr_logits.shape[1] == num_indcs_for_node_attribute_matrix

        # Now we want to put the edge attribute and adjacency matrices into a symmetric form
        # therefore we now create a matrix that can be used for mask and the empty adjacency and edge attribute matrices
        msk = torch.ones(batch_size, max_node_size, max_node_size, device=str(packed_tensor.device), dtype=torch.uint8)
        adj_mat = torch.zeros(batch_size, max_node_size, max_node_size, device=str(packed_tensor.device), dtype=packed_tensor.dtype)
        edge_attribute_tensor = torch.zeros(batch_size,  CHEM_DETAILS.num_bond_types, max_node_size, max_node_size,
                                            device=str(packed_tensor.device), dtype=packed_tensor.dtype)

        # Now add the correct entries in.
        adj_mat[msk.triu()] = adj_mat_logits.contiguous().view(-1)
        adj_mat += adj_mat.triu(diagonal=1).transpose(1,2)  # add the lower triagonal part.
        edge_attribute_tensor[msk[:, None, :, :].triu(diagonal=1).repeat(1,CHEM_DETAILS.num_bond_types, 1,1)] = edge_atr_logits.contiguous().view(-1)
        edge_attribute_tensor = edge_attribute_tensor.view(-1, max_node_size, max_node_size)
        edge_attribute_tensor += edge_attribute_tensor.triu(diagonal=1).transpose(1,2)
        edge_attribute_tensor = edge_attribute_tensor.view(batch_size,  CHEM_DETAILS.num_bond_types, max_node_size, max_node_size)
        edge_attribute_tensor = edge_attribute_tensor.permute(0,2,3,1)

        # Now reshape the node attribute
        node_atr_logits = node_atr_logits.view(batch_size, max_node_size, CHEM_DETAILS.num_node_types)

        return cls(adj_mat, edge_attribute_tensor, node_atr_logits)

