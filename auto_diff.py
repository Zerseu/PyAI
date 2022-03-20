import math

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


class UnaryOperator(Operator):
    def __init__(self, name='UnaryOperator'):
        super().__init__(name)

    def __repr__(self):
        return f'UnaryOperator: name:{self.name}'

    def forward(self, a):
        raise NotImplementedError()

    def backward(self, a, d_out):
        raise NotImplementedError()


class BinaryOperator(Operator):
    def __init__(self, name='BinaryOperator'):
        super().__init__(name)

    def __repr__(self):
        return f'BinaryOperator: name:{self.name}'

    def forward(self, a, b):
        raise NotImplementedError()

    def backward(self, a, b, d_out):
        raise NotImplementedError()


class Neg(UnaryOperator):
    count = 0

    def __init__(self, a, name=None):
        super().__init__(name)
        self.inputs = [a]
        self.name = f'Neg/{Neg.count}' if name is None else name
        Neg.count += 1

    def forward(self, a):
        return -a

    def backward(self, a, d_out):
        return [-d_out]


class Add(BinaryOperator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'Add/{Add.count}' if name is None else name
        Add.count += 1

    def forward(self, a, b):
        return a + b

    def backward(self, a, b, d_out):
        return [d_out, d_out]


class Sub(BinaryOperator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'Sub/{Sub.count}' if name is None else name
        Sub.count += 1

    def forward(self, a, b):
        return a - b

    def backward(self, a, b, d_out):
        return [d_out, -d_out]


class Mul(BinaryOperator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'Mul/{Mul.count}' if name is None else name
        Mul.count += 1

    def forward(self, a, b):
        return a * b

    def backward(self, a, b, d_out):
        return [d_out * b, d_out * a]


class Div(BinaryOperator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'Div/{Div.count}' if name is None else name
        Div.count += 1

    def forward(self, a, b):
        return a / b

    def backward(self, a, b, d_out):
        return [d_out / b, d_out * a / b ** 2]


class Pow(BinaryOperator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'Pow/{Pow.count}' if name is None else name
        Pow.count += 1

    def forward(self, a, b):
        return a ** b

    def backward(self, a, b, d_out):
        return [d_out * b * a ** (b - 1), d_out * math.log(a) * a ** b]


class MatMul(BinaryOperator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'MatMul/{MatMul.count}' if name is None else name
        MatMul.count += 1

    def forward(self, a, b):
        return a @ b

    def backward(self, a, b, d_out):
        return [d_out @ b.T, a.T @ d_out]


class Exp(UnaryOperator):
    count = 0

    def __init__(self, a, name=None):
        super().__init__(name)
        self.inputs = [a]
        self.name = f'Exp/{Exp.count}' if name is None else name
        Exp.count += 1

    def forward(self, a):
        return math.exp(a)

    def backward(self, a, d_out):
        return [d_out * math.exp(a)]


class Sin(UnaryOperator):
    count = 0

    def __init__(self, a, name=None):
        super().__init__(name)
        self.inputs = [a]
        self.name = f'Sin/{Sin.count}' if name is None else name
        Sin.count += 1

    def forward(self, a):
        return math.sin(a)

    def backward(self, a, d_out):
        return [d_out * math.cos(a)]


def node_wrapper_unary(func, self):
    return func(self)


def node_wrapper_binary(func, self, other):
    if isinstance(other, Node):
        return func(self, other)
    if isinstance(other, float):
        return func(self, Constant(other))
    raise TypeError('Incompatible types encountered!')


Node.neg = lambda self: node_wrapper_unary(Neg, self)
Node.add = lambda self, other: node_wrapper_binary(Add, self, other)
Node.sub = lambda self, other: node_wrapper_binary(Sub, self, other)
Node.mul = lambda self, other: node_wrapper_binary(Mul, self, other)
Node.div = lambda self, other: node_wrapper_binary(Div, self, other)
Node.pow = lambda self, other: node_wrapper_binary(Pow, self, other)
Node.matmul = lambda self, other: node_wrapper_binary(MatMul, self, other)
Node.exp = lambda self: node_wrapper_unary(Exp, self)
Node.sin = lambda self: node_wrapper_unary(Sin, self)


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
        x = Variable(0, name='x')
        o = Node.exp(Node.sin(Node.neg(x)))
        ordering = topological_sort(o)

        val = 1
        x.value = val
        forward_pass(ordering)
        backward_pass(ordering)

        dodx_node = [a for a in ordering if a.name == 'x'][0]
        print(f'do/dx expected = {-math.cos(val) * math.exp(-math.sin(val))}')
        print(f'do/dx computed = {dodx_node.gradient}')

        val = math.pi
        x.value = val
        forward_pass(ordering)
        backward_pass(ordering)

        dodx_node = [a for a in ordering if a.name == 'x'][0]
        print(f'do/dx expected = {-math.cos(val) * math.exp(-math.sin(val))}')
        print(f'do/dx computed = {dodx_node.gradient}')


if __name__ == "__main__":
    main()
