"""
Graph Convolutional Network in Tinygrad

This file implements https://arxiv.org/abs/1609.02907
Here is a good video explaining the topic: https://youtu.be/VyIOfIglrUM?si=QSMezKvcCL8cUDuF
"""
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear

class GraphConvolution:
  """
  Simple GCN layer
  """

  def __init__(self, in_features, out_features, bias=True):
    self.linear = Linear(in_features, out_features)

  def forward(self, x: Tensor, adjacency_hat: Tensor):
    x = self.linear(x)
    x = x.mul(adjacency_hat)
    return x
  
