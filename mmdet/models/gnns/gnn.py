from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph

class StateConvLayer(MessagePassing):
    def __init__(self, state_dim):
        """
        Graph Layer to perform update of the node states.
        """
        super(StateConvLayer, self).__init__(aggr='max')

        # MLP to transform node state into the relative offset to alleviate translation variance.
        self.mlp_h = Seq(Linear(state_dim, state_dim//2), 
                         ReLU(inplace=True),
                         Linear(state_dim//2, state_dim//4), 
                         ReLU(inplace=True),
                         Linear(state_dim//4, 3))

        # MLP to compute edge features
        self.mlp_f = Seq(Linear(state_dim+3, state_dim//2),
                         ReLU(inplace=True), 
                         Linear(state_dim//2, state_dim//4), 
                         ReLU(inplace=True),
                         Linear(state_dim//4, state_dim),
                         )

        self.mlp_g = Seq(Linear(state_dim, state_dim//2),
                         ReLU(inplace=True), 
                         Linear(state_dim//2, state_dim//4), 
                         ReLU(inplace=True),
                         Linear(state_dim//4,state_dim),
                         )

    def forward(self, s, x, edge_index):
        return self.propagate(edge_index, s=s, x=x)

    def message(self, x_j, x_i, s_i, s_j):

        # The extended graph update algorithm.
        delta_x_i = self.mlp_h(s_i)
        tmp = torch.cat([x_j - x_i - delta_x_i, s_j], dim=1)
        e_ij = self.mlp_f(tmp)
        return e_ij

    def update(self, e_ij, s):
        # Update vertex state based on aggregated edge features
        return s + self.mlp_g(e_ij)

def basic_block(in_channel, out_channel):
    """
    Create block with linear layer followed by IN and ReLU.
    :param in_channel: number of input features
    :param out_channel: number of output features
    :return: PyTorch Sequential object
    """
    return nn.Sequential(Linear(in_channel, out_channel),
                         nn.InstanceNorm1d(out_channel),
                         nn.ReLU(inplace=True))

class BARefiner(nn.Module):
    def __init__(self, state_dim, n_classes, n_iterations):
        """
        Boundary-Aware Graph Neural Network, which takes 3D proposals in immediate neighborhood
        as inputs for graph construction within a given cut-off distance, associating 3D proposals 
        in the form of local neighborhood graph, with boundary correlations of an object being 
        explicitly informed through an information compensation mechanism.

        Args:
        :param state_dim: maximum number of state features
        :param n_classes: number of classes
        :param n_iterations: number of GNN iterations to perform
        """
        super(BARefiner, self).__init__()
        self.state_dim = state_dim
        self.n_classes = n_classes
        self.n_iterations = n_iterations
        self._num_anchor_per_loc = 1
        self._box_code_size = 7

        # List of GNN layers
        self.graph_layers = nn.ModuleList([StateConvLayer(state_dim) for _ in
                                     range(n_iterations)])
        
        # MLP for class prediction
        self.mlp_class = Seq(basic_block(state_dim, state_dim),
                             basic_block(state_dim, state_dim),
                             Linear(state_dim, self._num_anchor_per_loc * self.n_classes))

        # Set of MLPs for per-class bounding box regression
        self.mlp_loc = nn.ModuleList([Seq(basic_block(state_dim, state_dim),
                                          basic_block(state_dim, state_dim),
                                          Linear(state_dim, self._num_anchor_per_loc * self._box_code_size)) for _ in
                                      range(self.n_classes)])

    def RadiusGraph(self, vertex):
         # Create graph
        return RadiusGraph(r=0.1, loop=True, max_num_neighbors=256,
                           flow='target_to_source')(Data(pos=vertex[...,1:4].float().contiguous(),batch=vertex[...,0].long()))

    def forward(self, batch_data):
        
        graph = self.RadiusGraph(batch_data['node_pos'])
        # Set initial vertex state
        state = batch_data['node_features']
        # Perform GNN computations
        for graph_layer in self.graph_layers:
            # Update vertex state
            state = graph_layer(state, graph.pos, graph.edge_index)
        state = state.unsqueeze(0)
        cls_pred = self.mlp_class(state)
        reg_pred = torch.cat([mlp_loc(state) for mlp_loc in self.mlp_loc], dim=2)
        return (reg_pred, cls_pred), None
