import inspect
from typing import Sequence, Union

import numpy as np

from ._shaped_value import ShapedValue, _indices_to_indexer
from .arith import Scalar, constant
from .tensor import compute_result_shape_reassoc_list
from .vector import Vector
from ... import types as T
from ...meta import region_op
from ...util import (
    _get_sym_name,
    get_user_code_loc,
    infer_mlir_type,
)
from ...._mlir_libs._mlir import register_value_caster
from ....dialects import memref, arith, vector, builtin
from ....dialects._ods_common import get_op_result_or_op_results
from ....dialects.memref import *
from ....ir import (
    DenseElementsAttr,
    MemRefType,
    ShapedType,
    Type,
    Value,
    SymbolTable,
    InsertionPoint,
)

S = ShapedType.get_dynamic_size()


def __alloc(
    op_ctor,
    sizes: Sequence[Union[int, Value]],
    element_type: Type,
    memory_space=None,
    alignment=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    dynamic_sizes = []
    memref_shape = []
    for s in sizes:
        if isinstance(s, int):
            memref_shape.append(s)
        else:
            memref_shape.append(ShapedType.get_dynamic_size())
            dynamic_sizes.append(s)
    result_type = T.memref(
        *memref_shape, element_type=element_type, memory_space=memory_space
    )

    symbol_operands = []
    return get_op_result_or_op_results(
        op_ctor(
            result_type,
            dynamic_sizes,
            symbol_operands,
            alignment=alignment,
            loc=loc,
            ip=ip,
        )
    )


_alloc = alloc


def alloc(
    sizes: Union[int, Value],
    element_type: Type = None,
    *,
    memory_space=None,
    alignment=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    return __alloc(
        AllocOp,
        sizes,
        element_type,
        memory_space=memory_space,
        alignment=alignment,
        loc=loc,
        ip=ip,
    )


_alloca = alloca


def alloca(
    sizes: Union[int, Value],
    element_type: Type = None,
    memory_space=None,
    alignment=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    return __alloc(
        AllocaOp,
        sizes,
        element_type,
        memory_space=memory_space,
        alignment=alignment,
        loc=loc,
        ip=ip,
    )


def load(memref: Value, indices: Sequence[Union[Value, int]], *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    indices = list(indices)
    for idx, i in enumerate(indices):
        if isinstance(i, int):
            indices[idx] = constant(i, index=True)
    return get_op_result_or_op_results(LoadOp(memref, indices, loc=loc, ip=ip))


def store(
    value: Value,
    memref: Value,
    indices: Sequence[Union[Value, int]],
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    indices = list(indices)
    for idx, i in enumerate(indices):
        if isinstance(i, int):
            indices[idx] = constant(i, index=True)
    return get_op_result_or_op_results(StoreOp(value, memref, indices, loc=loc, ip=ip))


@register_value_caster(MemRefType.static_typeid)
@ShapedValue
class MemRef(Value):
    def __str__(self):
        return f"{self.__class__.__name__}({self.get_name()}, {self.type})"

    def __repr__(self):
        return str(self)

    def __getitem__(self, idx: tuple) -> "MemRef":
        loc = get_user_code_loc()

        if not self.has_rank():
            raise ValueError("only ranked memref slicing/indexing supported")

        if idx == Ellipsis or idx == slice(None):
            return self
        if isinstance(idx, tuple) and all(i == slice(None) for i in idx):
            return self
        if idx is None:
            return expand_shape(self, (0,), loc=loc)

        idx = list((idx,) if isinstance(idx, (int, Scalar, slice)) else idx)
        for i, d in enumerate(idx):
            # TODO(max): rethink this since subview and etc probably take constant attributes?
            if isinstance(d, int):
                idx[i] = constant(d, index=True, loc=loc)

        if all(isinstance(d, Scalar) for d in idx) and len(idx) == len(self.shape):
            return load(self, idx, loc=loc)
        else:
            return _subview(self, tuple(idx), loc=loc)

    def __setitem__(self, idx, val):
        loc = get_user_code_loc()

        if not self.has_rank():
            raise ValueError("only ranked memref slicing/indexing supported")

        idx = list((idx,) if isinstance(idx, (Scalar, int, Value)) else idx)
        for i, d in enumerate(idx):
            if isinstance(d, int):
                idx[i] = constant(d, index=True, loc=loc)

        if all(isinstance(d, Scalar) for d in idx) and len(idx) == len(self.shape):
            if isinstance(val, (int, float)):
                # TODO: this is an unchecked conversion
                val = Scalar(val, dtype=self.dtype)
            assert isinstance(
                val, (Scalar, Vector)
            ), "coordinate insert requires scalar element"
            if isinstance(val, Scalar):
                store(val, self, idx, loc=loc)
            elif isinstance(val, Vector):
                return vector.StoreOp(
                    valueToStore=val,
                    base=self,
                    indices=idx,
                    loc=loc,
                )
        else:
            _copy_to_subview(self, val, tuple(idx), loc=loc)


def expand_shape(
    inp,
    newaxis_dims,
    *,
    loc=None,
    ip=None,
) -> MemRef:
    """Expand the shape of a memref.

    Insert a new axis that will appear at the `axis` position in the expanded
    memref shape.

    Args:
      inp: Input memref-like.
      axis: Position in the expanded axes where the new axis (or axes) is placed.

    Returns:
       View of `a` with the number of dimensions increased.

    """
    if loc is None:
        loc = get_user_code_loc()

    if len(newaxis_dims) == 0:
        return inp

    result_shape, reassoc_list = compute_result_shape_reassoc_list(
        inp.shape, newaxis_dims
    )

    return MemRef(
        memref.expand_shape(
            T.memref(*result_shape, inp.dtype),
            inp,
            reassoc_list,
            output_shape=[],
            static_output_shape=result_shape,
            loc=loc,
            ip=ip,
        )
    )


def _canonicalize_start_stop(start, stop, step):
    # TODO(max): figure out how to use actual canonicalizers
    if (
        isinstance(start, Value)
        and isinstance(stop, Value)
        and stop.owner.operands[0]._eq(start)
        and stop.owner.operands[1].is_constant()
    ):
        return stop.owner.operands[1].literal_value
    elif (
        isinstance(start.owner.opview, arith.MulIOp)
        and isinstance(stop.owner.opview, arith.MulIOp)
        and isinstance(stop.owner.operands[0].owner.opview, arith.AddIOp)
        and start.owner.operands[0] == stop.owner.operands[0].owner.operands[0]
        and stop.owner.operands[1].is_constant()
        and isinstance(step, int)
        or isinstance(step, Scalar)
        and step.is_constant()
    ):
        # looks like this
        # l = lambda l: l * D
        # r = lambda r: (r + 1) * D
        # a, b, c = (
        #     A[l(i) : r(i), l(j) : r(j)],
        #     B[l(i) : r(i), l(j) : r(j)],
        #     C[l(i) : r(i), l(j) : r(j)],
        # )
        return stop.owner.operands[1]
    elif isinstance(start, int) and isinstance(stop, int):
        return stop - start


def _subview(
    mem: MemRef,
    idx,
    *,
    loc=None,
    ip=None,
) -> MemRef:
    if loc is None:
        loc = get_user_code_loc()

    indexer = _indices_to_indexer(idx, mem.shape)
    out = mem

    if indexer.is_constant():
        out = subview(
            out,
            offsets=indexer.static_offsets(),
            sizes=indexer.static_sizes(),
            strides=indexer.static_strides(),
            loc=loc,
            ip=ip,
        )
    else:
        # special tile case
        offsets = [None] * len(indexer.in_shape)
        static_sizes = [None] * len(indexer.in_shape)
        static_strides = [None] * len(indexer.in_shape)
        for i, ind in enumerate(indexer.indices):
            maybe_size = _canonicalize_start_stop(ind.start, ind.stop, ind.step)
            if maybe_size is not None:
                offsets[i] = ind.start
                static_sizes[i] = maybe_size
                static_strides[i] = (
                    ind.step.literal_value if isinstance(ind.step, Scalar) else ind.step
                )
            else:
                raise RuntimeError(f"indexing not supported {indexer.indices}")
        assert all(
            map(lambda x: x is not None, offsets + static_sizes + static_strides)
        ), f"not each slice is statically known: {indexer.indices}"
        out = subview(
            out,
            offsets=offsets,
            sizes=static_sizes,
            strides=static_strides,
            loc=loc,
            ip=ip,
        )

    # This adds newaxis/None dimensions.
    return expand_shape(out, indexer.newaxis_dims, loc=loc, ip=ip)


def _copy_to_subview(
    dest: MemRef,
    source: MemRef,
    idx,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(source, Scalar):
        source = expand_shape(source, (0,), loc=loc, ip=ip)

    dest_subview = _subview(dest, idx, loc=loc, ip=ip)
    assert (
        dest_subview.shape == source.shape
    ), f"Expected matching shape for dest subview {dest_subview.shape} and source {source.shape=}"

    return memref.copy(source, dest_subview, loc=loc, ip=ip)


alloca_scope = region_op(AllocaScopeOp)

_dim = dim


def dim(source, index, *, loc=None, ip=None):
    if isinstance(index, int):
        index = constant(index, index=True)
    return _dim(source=source, index=index, loc=loc, ip=ip)


def global_(
    initial_value=None,
    sym_name=None,
    type=None,
    sym_visibility="private",
    constant=None,
    alignment=None,
    loc=None,
    ip=None,
):
    if sym_name is None:
        previous_frame = inspect.currentframe().f_back
        sym_name = _get_sym_name(
            previous_frame, check_func_call="memref\\.global_|global_"
        )
        assert (
            sym_name is not None
        ), "couldn't automatically find sym_name in previous frame"
    if loc is None:
        loc = get_user_code_loc()
    if initial_value is None:
        assert type is not None
    else:
        assert isinstance(initial_value, np.ndarray)
        if type is None:
            type = infer_mlir_type(initial_value, memref=True)
        initial_value = DenseElementsAttr.get(
            initial_value,
            type=type.element_type,
            context=None,
        )
        constant = True

    return memref.global_(
        sym_name,
        type,
        sym_visibility=sym_visibility,
        initial_value=initial_value,
        constant=constant,
        alignment=alignment,
        loc=loc,
        ip=ip,
    ).opview


def view(source, shape, dtype=None, shift=0, memory_space=None):
    if dtype is None:
        dtype = source.type.element_type
    byte_width_dtype = dtype.width // 8
    byte_shift = shift * byte_width_dtype
    byte_shift = constant(byte_shift, index=True)
    if memory_space is None and source:
        memory_space = source.type.memory_space
    return memref.view(
        T.memref(*shape, element_type=dtype, memory_space=memory_space),
        source,
        byte_shift,
        [],
    )


_get_global = get_global


def get_global(
    name_or_global, *, name=None, global_=None, result=None, loc=None, ip=None
):
    if isinstance(name_or_global, GlobalOp):
        global_ = name_or_global
    elif isinstance(name_or_global, str):
        name = name_or_global
    elif name_or_global is not None:
        raise ValueError(
            f"only string or GlobalOp can be provided; got {name_or_global}"
        )

    if global_ is None:
        assert name is not None, "name must be provided"

        if result is None:

            def callback(symbol_table_op, _uses_visible):
                nonlocal global_
                sym_table = SymbolTable(symbol_table_op)
                if name in sym_table:
                    global_ = sym_table[name]

            current_owner = InsertionPoint.current.block.owner
            while not isinstance(current_owner.opview, builtin.ModuleOp):
                current_owner = current_owner.parent
            SymbolTable.walk_symbol_tables(current_owner, True, callback)
            if global_ is None:
                raise RuntimeError(f"couldn't find symbol for {name}")

    if not isinstance(global_, GlobalOp):
        raise RuntimeError(f"expected memref.global, got {global_}")
    result = global_.type_.value
    name = global_.sym_name.value
    return GetGlobalOp(result=result, name=name, loc=loc, ip=ip).result
