'''
Author: Diantao Tu
Date: 2024-01-22 18:50:20
'''
from graphviz import Digraph
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from typing import Dict, List

from encoder import ResNet50Autoencoder

'''
if the input dict is a nested dict, then flatten it to a single level dict.
for example:
input_dict = {
    'layer1': {
        'layer2': {
            'layer3': {
               }
            }
        }
    }
becomes {'layer1.layer2.layer3': {}}
'''
def combine_dict(input_dict:Dict) -> Dict:
    output_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, dict):
            sub_dict = combine_dict(v)
            for sub_k, sub_v in sub_dict.items():
                output_dict[k + '.' + sub_k] = sub_v
        else:
            output_dict[k] = v
    return output_dict

def add_val_to_dict(input_dict:Dict, key, val):
    if key in input_dict:
        input_dict[key].append(val)
    else:
        input_dict[key] = [val]

class ComputeGraph:
    node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
    
    # Saved attrs for grad_fn (incl. saved variables) begin with `._saved_*`
    SAVED_PREFIX = "_saved_"

    NORMAL_GRAD = 0
    NAN_GRAD = 1
    INF_GRAD = 2
    LOW_GRAD = 3
    HIGH_GRAD = 4
    
    '''
    params: dict, key = name, value = value. Used to add names to the nodes
            of the graph. If params is None, then the nodes will be named empty.
    max_attr_length: int, maximum length of the attribute string to display.
            If the length of the attribute string is greater than this value,
            the attribute string will be truncated. Only effective when 
            add_backward_fn_attrs is called.
    grad_upper_bound: float, the upper bound of the gradient. If the gradient
            is greater than this value, the node is colored pink.
    grad_lower_bound: float, the lower bound of the gradient. If the gradient
            is less than this value, the node is colored pink.
    '''
    def __init__(self, params=None, max_attr_length=20, grad_upper_bound=1e4, grad_lower_bound=-1e4) -> None:
        self.fn_dict = {}
        self.graph = Digraph(node_attr=self.node_attr, graph_attr=dict(size="120,120"))
        self.edge_set = set()
        self.param_map = {}
        if params is not None:
            params = combine_dict(params)
            self.param_map = {id(v): k for k, v in params.items()}
        self.max_attr_length = max_attr_length
        self.grad_upper_bound = grad_upper_bound
        self.grad_lower_bound = grad_lower_bound
        

    def clear(self) -> None:
        self.fn_dict = {}
        self.edge_set = set()
        self.saved_tensor_to_backward_fn = {}
        self.backward_fn_attrs = {}
        self.graph = Digraph(node_attr=self.node_attr, graph_attr=dict(size="120,120"))

    def register_hooks(self, var):
        def hook_cb(fn):
            def register_grad(grad_input, grad_output):
                self.fn_dict[fn] = grad_input
            fn.register_hook(register_grad)
        
        self._iter_graph(var.grad_fn, [hook_cb])

    def add_saved_tensors(self, var):
        # When calling add_saved_tensors function, the graph is not built yet, 
        # so we just create the node for the variable here.
        # The node will be connected to the backward node later.
        self.saved_tensor_to_backward_fn = {}
        self._iter_graph(var.grad_fn, [self._add_saved_tensors])

    def add_backward_fn_attrs(self, var):
        # When calling add_backward_fn_attrs function, the graph is not built yet,
        # so we just get the attributes of the backward node here.
        # The attributes will be added to the backward node later.
        self.backward_fn_attrs = {}
        self._iter_graph(var.grad_fn, [self._add_backward_fn_attrs])

    def get_dot(self, var):
        
        self._iter_graph(var.grad_fn, [self._build_graph])
        if hasattr(self, 'saved_tensor_to_backward_fn'):
            for saved_tensor_id, backward_fn_ids in self.saved_tensor_to_backward_fn.items():
                for backward_fn_id in backward_fn_ids:
                    self._add_edge(saved_tensor_id, backward_fn_id, dir="none")

        self._resize_graph()
        if hasattr(self, 'backward_fn_attrs') and  len(self.backward_fn_attrs) > 0:
            self._resize_graph(0.4)

        return self.graph

    def _is_bad_grad(self, grad_output) -> int:
        if grad_output is None:
            return False
        grad_output = grad_output.data

    
        if torch.isnan(grad_output).any():
            return self.NAN_GRAD
        if torch.isinf(grad_output).any():
            return self.INF_GRAD
        if grad_output.gt(self.grad_upper_bound).any():
            return self.HIGH_GRAD
        if grad_output.lt(self.grad_lower_bound).any():
            return self.LOW_GRAD
    
    def _size_to_str(self, size):
        return '('+(', ').join(map(str, size))+')'
    
    def _add_edge(self, u, v, **kwargs):
        if ((u, v) not in self.edge_set) and (u != v) :
            self.edge_set.add((u, v))
            self.graph.edge(str(u), str(v), **kwargs)


    def _get_fn_name(self, fn):
        name = str(type(fn).__name__)
       
        if not hasattr(self, 'backward_fn_attrs'):
            return name
       
        attrs = self.backward_fn_attrs.get(id(fn), {})
        if not attrs:
            return name
        
        max_attr_length = max(self.max_attr_length, 3)
        col1_width = max(len(k) for k in attrs.keys())
        col2_width = min(max(len(v) for v in attrs.values()), max_attr_length)
        sep = "-" * max(col1_width + col2_width + 2, len(name))
        attrstr = '%-' + str(col1_width) + 's: %' + str(col2_width)+ 's'
        truncate = lambda s: s[:col2_width - 3] + "..." if len(s) > col2_width else s
        params = '\n'.join(attrstr % (k, truncate(str(v))) for (k, v) in attrs.items())
        return name + '\n' + sep + '\n' + params


    def _build_graph(self, fn):
        if hasattr(fn, 'variable'):
            u = fn.variable
            name = self.param_map.get(id(u), '')
            node_name = name + '\n' + self._size_to_str(u.size())
            self.graph.node(str(id(u)), node_name, fillcolor='lightblue')
        else:
            fillcolor = 'white'
            # 在某些情况下, 前向与反向的计算图不一致, 这是由于 ReLU 或者 BatchNorm 或者 Dropout 等操作的原因
            # 比如 ReLU 对负数输入结果是 0, 因此在前向时就不会有相关的计算图
            # 但是在反向时, 会有计算图, 因此在这种情况下, 会导致前向与反向的计算图不一致
            if fn in self.fn_dict:
                grad_type = [self._is_bad_grad(gi) for gi in self.fn_dict[fn]]
                if self.NAN_GRAD in grad_type or self.INF_GRAD in grad_type:
                    fillcolor = 'red'
                elif self.LOW_GRAD in grad_type or self.HIGH_GRAD in grad_type:
                    fillcolor = 'pink'
            self.graph.node(str(id(fn)), self._get_fn_name(fn), fillcolor=fillcolor)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                next_id = id(getattr(next_fn, 'variable', next_fn))
                self._add_edge(next_id, id(fn))
     

    def _add_saved_tensors(self, fn):
        
        for attr in dir(fn):
            if not attr.startswith(self.SAVED_PREFIX):
                continue
            val = getattr(fn, attr)
            attr = attr[len(self.SAVED_PREFIX):]
            if isinstance(val, torch.Tensor):
                add_val_to_dict(self.saved_tensor_to_backward_fn, id(val), id(fn))
                
                name = attr + '\n' + self._size_to_str(val.size())
                self.graph.node(str(id(val)), name, fillcolor='orange')
            if isinstance(val, tuple):
                for i, v in enumerate(val):
                    if not isinstance(v, torch.Tensor):
                        continue
                    add_val_to_dict(self.saved_tensor_to_backward_fn, id(v), id(fn))
                    name = attr + '.' + str(i) + '\n' + self._size_to_str(v.size())
                    self.graph.node(str(id(v)), name, fillcolor='orange')

    def _add_backward_fn_attrs(self, fn):
        attrs = {}
        for attr in dir(fn):
            if not attr.startswith(self.SAVED_PREFIX):
                continue
            val = getattr(fn, attr)
            attr = attr[len(self.SAVED_PREFIX):]
            if isinstance(val, torch.Tensor):
                attrs[attr] = "[saved tensor]"
            elif isinstance(val, tuple) and all(isinstance(v, torch.Tensor) for v in val):
                attrs[attr] = "[saved {} tensors]".format(len(val))
            else:
                attrs[attr] = str(val)
        self.backward_fn_attrs[id(fn)] = attrs

    def _iter_graph(self, root, callback_list:List):
        queue = [root]
        seen = set()
        while queue:
            fn = queue.pop()
            if fn in seen:
                continue
            seen.add(fn)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    queue.append(next_fn)
            for callback in callback_list:
                callback(fn)

    def _resize_graph(self, size_per_element=0.2, min_size=12):
        """Resize the graph according to how much content it contains.

        Modify the graph in place.
        """
        # Get the approximate number of nodes and edges
        num_rows = len(self.graph.body)
        content_size = num_rows * size_per_element
        size = max(min_size, content_size)
        size_str = str(size) + "," + str(size)
        self.graph.graph_attr.update(size=size_str)

if __name__ == '__main__':

    

    x = Variable(torch.randn(10, 10), requires_grad=True)
    y = Variable(torch.randn(10, 10), requires_grad=True)
    compute_graph = ComputeGraph({'x': x, 'y': y})

    z = x / (y * 0)
    z = z.sum() * 2
    
    compute_graph.register_hooks(z)
    compute_graph.add_saved_tensors(z)
    compute_graph.add_backward_fn_attrs(z)

    z.backward()
    dot = compute_graph.get_dot(z)

    dot.format = 'png'
    dot.render('visualize')


    

