<!--
 * @Author: Diantao Tu
 * @Date: 2024-01-23 18:07:59
-->
# Pytorch Compute Graph Visualization

This is a simple tool to visualize the compute graph of pytorch model. 
It is based on [pytorchviz](https://github.com/szagoruyko/pytorchviz/tree/master) and [this](https://gist.github.com/apaszke/f93a377244be9bfcb96d3547b9bc424d).

## Usage

```python
from compute_graph import ComputeGraph 

model = Model()     # Your pytorch model
compute_graph = ComputeGraph(dict(model.named_parameters()))

x = ...        # Your input tensor
y = model(x)   # Your output tensor
loss = loss_function()      # Your loss function

compute_graph.register_hooks(loss)
compute_graph.add_saved_tensors(loss)
compute_graph.add_backward_fn_attrs(loss)

loss.backward()

dot = compute_graph.get_dot(z)
dot.format = 'png'
dot.render('visualize')
```
For more details, please refer to the [example](example.ipynb).

## Environment

The visualization is based on [graphviz](https://graphviz.org/). 
Please install it first.
```bash
sudo apt-get install graphviz
```
or 
```bash
conda install graphviz
conda install python-graphviz
```

**Do not install graphviz with pip.**