"""Index projection for the existing CC density expression, not new equations.

Only its explicit einsums, scalar arithmetic, transposes/conjugates and fixed
O/V scatter segments are supported. Caches belong to one evaluated tile and
hold no values across calls. Tensor/segment objects are never JAX array inputs.
"""
import jax.numpy as jnp


class Index:
    """Identity-keyed axis selection; its values remain ordinary JAX inputs."""
    def __init__(self, values):
        self.values = values


class Tensor:
    def __init__(self, value):
        self.shape = getattr(value, 'shape', ())
        self.operation, self.arguments, self.option = 'value', (value,), None

    @classmethod
    def expression(cls, operation, shape, arguments, option=None):
        node = cls.__new__(cls)
        node.operation, node.shape = operation, tuple(shape)
        node.arguments, node.option = tuple(arguments), option
        return node

    def _binary(self, operation, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if other.shape and other.shape != self.shape:
            raise ValueError('Density expressions support equal-shape or scalar arithmetic')
        return self.expression(operation, self.shape, (self, other))

    def __add__(self, other): return self._binary('add', other)
    def __sub__(self, other): return self._binary('sub', other)
    def __mul__(self, other): return self._binary('mul', other)
    __radd__, __rmul__ = __add__, __mul__
    def __neg__(self): return self * -1
    def conj(self): return self.expression('conj', self.shape, (self,))

    def transpose(self, *axes):
        return self.expression('transpose', tuple(self.shape[i] for i in axes), (self,), tuple(axes))

    def evaluate(self, selection, cache):
        key = self, tuple(selection)
        if key in cache:
            return cache[key]
        operation, arguments = self.operation, self.arguments
        if operation == 'value':
            value = arguments[0]
            for axis, index in enumerate(selection):
                if index is not None:
                    value = jnp.take(value, index.values, axis=axis)
        elif operation == 'einsum':
            inputs, output = self.option.split('->')
            selected = dict(zip(output, selection))
            values = [node.evaluate(tuple(selected.get(label) for label in labels), cache)
                      for labels, node in zip(inputs.split(','), arguments)]
            value = jnp.einsum(self.option, *values)
        elif operation == 'transpose':
            original = [None]*len(selection)
            for axis, source in enumerate(self.option):
                original[source] = selection[axis]
            value = arguments[0].evaluate(tuple(original), cache).transpose(self.option)
        elif operation == 'conj':
            value = arguments[0].evaluate(selection, cache).conj()
        else:
            left, right = (node.evaluate(selection if node.shape else (), cache) for node in arguments)
            if operation == 'add': value = left+right
            elif operation == 'sub': value = left-right
            elif operation == 'mul': value = left*right
            else: raise NotImplementedError(operation)
        cache[key] = value
        return value


def einsum(equation, *values):
    nodes = tuple(value if isinstance(value, Tensor) else Tensor(value) for value in values)
    inputs, output = equation.split('->')
    dimensions = {}
    for labels, node in zip(inputs.split(','), nodes):
        for label, size in zip(labels, node.shape):
            if label in dimensions and dimensions[label] != size:
                raise ValueError('Inconsistent CC density contraction dimensions')
            dimensions[label] = size
    return Tensor.expression('einsum', tuple(dimensions[label] for label in output), nodes, equation)


class _Setter:
    def __init__(self, buffer, slices=None): self.buffer, self.slices = buffer, slices
    def __getitem__(self, slices): return _Setter(self.buffer, slices)
    def set(self, value):
        b = self.buffer
        return Buffer(b.dimensions, b.value_dtype, b.segments+((self.slices, value),), b.permutation)


class Buffer:
    """The original sixteen disjoint O/V assignments without a dense array."""
    def __init__(self, dimensions, dtype, segments=(), permutation=(0,1,2,3)):
        # Avoid shape/dtype duck typing: this object owns no logical n**4 array.
        self.dimensions, self.value_dtype = tuple(dimensions), dtype
        self.segments, self.permutation = segments, permutation

    @property
    def at(self): return _Setter(self)

    def transpose(self, *axes):
        return Buffer(self.dimensions, self.value_dtype, self.segments,
                      tuple(self.permutation[i] for i in axes))

    def take(self, indices):
        base = tuple(indices[self.permutation.index(i)] for i in range(4))
        sizes = tuple(len(index) if isinstance(index, tuple) else index.shape[0] for index in base)
        result = jnp.zeros(sizes, self.value_dtype)
        selectors, cache = {}, {}
        for slices, expression in self.segments:
            chosen, positions, masks = [], [], []
            for axis, (selection, section) in enumerate(zip(base, slices)):
                start, stop, step = section.indices(self.dimensions[axis])
                if step != 1:
                    raise NotImplementedError('Density segments require contiguous O/V slices')
                key = axis, start, stop
                if key not in selectors:
                    if isinstance(selection, tuple):
                        places = tuple(i for i, value in enumerate(selection) if start <= value < stop)
                        values = jnp.asarray(tuple(selection[i]-start for i in places), jnp.int32)
                        entry = Index(values), jnp.asarray(places, jnp.int32), None
                    else:
                        values = jnp.clip(selection-start, 0, max(0, stop-start-1))
                        entry = Index(values), jnp.arange(sizes[axis]), (selection >= start) & (selection < stop)
                    selectors[key] = entry
                index, places, mask = selectors[key]
                if start == stop or places.size == 0:
                    break
                chosen.append(index); positions.append(places); masks.append(mask)
            else:
                value = expression.evaluate(tuple(chosen), cache).astype(self.value_dtype)
                for axis, mask in enumerate(masks):
                    if mask is not None:
                        shape = tuple(sizes[axis] if k == axis else 1 for k in range(4))
                        value = jnp.where(mask.reshape(shape), value, 0.)
                result = result.at[jnp.ix_(*positions)].add(value)
        return result.transpose(self.permutation)
