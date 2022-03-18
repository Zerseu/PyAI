import numpy as np

session = None


class Graph:
    def __init__(self):
        self.operators = set()
        self.constants = set()
        self.variables = set()
        self.placeholders = set()
        global session
        session = self

    def reset_counts(self, root):
        if hasattr(root, 'count'):
            root.count = 0
        else:
            for child in root.__subclasses__():
                self.reset_counts(child)

    def reset_session(self):
        try:
            del session
        except:
            pass
        self.reset_counts(Node)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reset_session()


class Node:
    def __init__(self):
        pass


class Placeholder(Node):
    count = 0

    def __init__(self, name):
        super().__init__()
        session.placeholders.add(self)
        self.value = None
        self.gradient = None
        self.name = f'Plc/{Placeholder.count}' if name is None else name
        Placeholder.count += 1

    def __repr__(self):
        return f'Placeholder: name:{self.name}, value:{self.value}'


class Constant(Node):
    count = 0

    def __init__(self, value, name=None):
        super().__init__()
        session.constants.add(self)
        self._value = value
        self.gradient = None
        self.name = f'Const/{Constant.count}' if name is None else name
        Constant.count += 1

    def __repr__(self):
        return f'Constant: name:{self.name}, value:{self.value}'

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        raise ValueError('Cannot reassign constant!')


class Variable(Node):
    count = 0

    def __init__(self, value, name=None):
        super().__init__()
        session.variables.add(self)
        self.value = value
        self.gradient = None
        self.name = f'Var/{Variable.count}' if name is None else name
        Variable.count += 1

    def __repr__(self):
        return f'Variable: name:{self.name}, value:{self.value}'


class Operator(Node):
    def __init__(self, name='Operator'):
        super().__init__()
        session.operators.add(self)
        self.value = None
        self.inputs = []
        self.gradient = None
        self.name = name

    def __repr__(self):
        return f'Operator: name:{self.name}'

    def forward(self, a, b):
        raise NotImplementedError()

    def backward(self, a, b, d_out):
        raise NotImplementedError()


class Add(Operator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'add/{Add.count}' if name is None else name
        Add.count += 1

    def forward(self, a, b):
        return a + b

    def backward(self, a, b, d_out):
        return d_out, d_out


class Sub(Operator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'sub/{Sub.count}' if name is None else name
        Sub.count += 1

    def forward(self, a, b):
        return a - b

    def backward(self, a, b, d_out):
        return d_out, -d_out


class Mul(Operator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'mul/{Mul.count}' if name is None else name
        Mul.count += 1

    def forward(self, a, b):
        return a * b

    def backward(self, a, b, d_out):
        return d_out * b, d_out * a


class Div(Operator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'div/{Div.count}' if name is None else name
        Div.count += 1

    def forward(self, a, b):
        return a / b

    def backward(self, a, b, d_out):
        return d_out / b, d_out * a / np.power(b, 2)


class Pow(Operator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'pow/{Pow.count}' if name is None else name
        Pow.count += 1

    def forward(self, a, b):
        return np.power(a, b)

    def backward(self, a, b, d_out):
        return d_out * b * np.power(a, (b - 1)), d_out * np.log(a) * np.power(a, b)


class MatMul(Operator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'matmul/{MatMul.count}' if name is None else name
        MatMul.count += 1

    def forward(self, a, b):
        return a @ b

    def backward(self, a, b, d_out):
        return d_out @ b.T, a.T @ d_out


def node_wrapper(func, self, other):
    if isinstance(other, Node):
        return func(self, other)
    if isinstance(other, float) or isinstance(other, int):
        return func(self, Constant(other))
    raise TypeError('Incompatible types encountered!')


Node.__neg__ = lambda self: node_wrapper(Mul, self, Constant(-1))
Node.__add__ = lambda self, other: node_wrapper(Add, self, other)
Node.__sub__ = lambda self, other: node_wrapper(Sub, self, other)
Node.__mul__ = lambda self, other: node_wrapper(Mul, self, other)
Node.__truediv__ = lambda self, other: node_wrapper(Div, self, other)
Node.__pow__ = lambda self, other: node_wrapper(Pow, self, other)
Node.__matmul__ = lambda self, other: node_wrapper(MatMul, self, other)


def topological_sort(head_node=None, graph=session):
    visiting = set()
    ordering = []

    def dfs(crt_node):
        if crt_node not in visiting:
            visiting.add(crt_node)
            if isinstance(crt_node, Operator):
                for input_node in crt_node.inputs:
                    dfs(input_node)
            ordering.append(crt_node)

    if head_node is None:
        for node in graph.operators:
            dfs(node)
    else:
        dfs(head_node)
    return ordering


def forward_pass(ordering, feed_dict=None):
    for node in ordering:
        if isinstance(node, Placeholder):
            node.value = feed_dict[node.name]
        elif isinstance(node, Operator):
            node.value = node.forward(*[prev_node.value for prev_node in node.inputs])
    return ordering[-1].value


def backward_pass(ordering):
    visiting = set()
    ordering[-1].gradient = 1
    for node in reversed(ordering):
        if isinstance(node, Operator):
            inputs = node.inputs
            grads = node.backward(*[x.value for x in inputs], d_out=node.gradient)
            for inp, grad in zip(inputs, grads):
                if inp not in visiting:
                    inp.gradient = grad
                else:
                    inp.gradient += grad
                visiting.add(inp)
    return [node.gradient for node in ordering]


def main():
    with Graph() as _:
        val1, val2, val3 = 0.9, 0.4, 1.3

        x = Variable(val1, name='x')
        y = Variable(val2, name='y')
        c = Constant(val3, name='c')
        z = x - y + c

        ordering = topological_sort(z)
        forward_pass(ordering)
        backward_pass(ordering)

        dzdx_node = [a for a in ordering if a.name == 'x'][0]
        dzdy_node = [a for a in ordering if a.name == 'y'][0]
        dzdc_node = [a for a in ordering if a.name == 'c'][0]

        print(f'dz/dx expected = {1}')
        print(f'dz/dx computed = {dzdx_node.gradient}')

        print(f'dz/dy expected = {-1}')
        print(f'dz/dy computed = {dzdy_node.gradient}')

        print(f'dz/dc expected = {1}')
        print(f'dz/dc computed = {dzdc_node.gradient}')


if __name__ == "__main__":
    main()
