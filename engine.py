import math
import random


class Module:
    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0

    def parameters(self):
        return []


class Value:
    def __init__(self, data, children=[], op="", label=""):
        self.data = data
        self.children = children
        self.op = op
        self.label = label
        self.parent = self
        self.grad = 0.0

        for child in self.children:
            child.parent = self

    def __repr__(self):
        return f"Value({self.label} = {self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, [self, other], "+")

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, [self, other], "*")

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        return Value(t, [self], "tanh")

    def backward(self, gradient=1.0):
        self.grad += gradient
        if self.op == "+":
            self.children[0].backward(gradient)
            self.children[1].backward(gradient)
        elif self.op == "*":
            self.children[0].backward(gradient * self.children[1].data)
            self.children[1].backward(gradient * self.children[0].data)
        elif self.op == "tanh":
            x = self.children[0].data
            t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
            self.children[0].backward(gradient * (1 - t * t))

    # Topological Sort not needed since it is already implemented in backward function
    # Backward function is called from parent to children recursively
    def topo_sort(self, visited=None, topo=None):
        if visited is None:
            visited = set()
        if topo is None:
            topo = []

        if self not in visited:
            visited.add(self)
            for child in self.children:
                child.topo_sort(visited, topo)
            topo.append(self.label)
        return topo

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1


class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"tanh Neuron - {[i.data for i in self.parameters()]}"


class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
