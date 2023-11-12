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
    x = x @ adjacency_hat
    return x
  
class GCN:
  def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=0, dropout=0.1, residual=False):
    super(GCN, self).__init__()

    self.dropout = dropout
    self.residual = residual

    self.input_conv = GraphConvolution(input_size, hidden_size)
    self.hidden_convs = [GraphConvolution(hidden_size, hidden_size) for _ in range(num_hidden_layers)]
    self.output_conv = GraphConvolution(hidden_size, output_size)

  def forward(self, x: Tensor, adjacency_hat: Tensor, labels: Tensor = None):
    x = x.dropout(p=self.dropout, training=self.training)
    x = self.input_conv(x, adjacency_hat).relu()

    for conv in self.hidden_convs:
      if self.residual:
        x = conv(x, adjacency_hat).relu() + x
      else:
        x = conv(x, adjacency_hat).relu()

    x = x.dropout(p=self.dropout, training=self.training)
    x = self.output_conv(x, adjacency_hat)

    if labels is None:
      return x

    loss = x.sparse_categorical_crossentropy(labels)
    return x, loss

