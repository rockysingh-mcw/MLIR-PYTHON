
# Autogenerated by mlir-tblgen; don't manually edit.

from ._ods_common import _cext as _ods_cext
from ._ods_common import (
    equally_sized_accessor as _ods_equally_sized_accessor,
    get_default_loc_context as _ods_get_default_loc_context,
    get_op_result_or_op_results as _get_op_result_or_op_results,
    get_op_results_or_values as _get_op_results_or_values,
    segmented_accessor as _ods_segmented_accessor,
)
_ods_ir = _ods_cext.ir

import builtins
from typing import Sequence as _Sequence, Union as _Union


@_ods_cext.register_dialect
class _Dialect(_ods_ir.Dialect):
  DIALECT_NAMESPACE = "bufferization"

@_ods_cext.register_operation(_Dialect)
class AllocTensorOp(_ods_ir.OpView):
  OPERATION_NAME = "bufferization.alloc_tensor"

  _ODS_OPERAND_SEGMENTS = [-1,0,0,]

  _ODS_REGIONS = (0, True)

  def __init__(self, result, dynamic_sizes, *, copy=None, size_hint=None, memory_space=None, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_results_or_values(dynamic_sizes))
    operands.append(copy)
    operands.append(size_hint)
    _ods_context = _ods_get_default_loc_context(loc)
    if memory_space is not None: attributes["memory_space"] = (memory_space if (
        isinstance(memory_space, _ods_ir.Attribute) or
        not _ods_ir.AttrBuilder.contains('AnyAttr')) else
          _ods_ir.AttrBuilder.get('AnyAttr')(memory_space, context=_ods_context))
    results.append(result)
    _ods_successors = None
    super().__init__(self.OPERATION_NAME, self._ODS_REGIONS, self._ODS_OPERAND_SEGMENTS, self._ODS_RESULT_SEGMENTS, attributes=attributes, results=results, operands=operands, successors=_ods_successors, regions=regions, loc=loc, ip=ip)

  @builtins.property
  def dynamic_sizes(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operandSegmentSizes"], 0)
    return operand_range

  @builtins.property
  def copy(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operandSegmentSizes"], 1)
    return operand_range[0] if len(operand_range) > 0 else None

  @builtins.property
  def size_hint(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operandSegmentSizes"], 2)
    return operand_range[0] if len(operand_range) > 0 else None

  @builtins.property
  def memory_space(self):
    if "memory_space" not in self.operation.attributes:
      return None
    return self.operation.attributes["memory_space"]

  @memory_space.setter
  def memory_space(self, value):
    if value is not None:
      self.operation.attributes["memory_space"] = value
    elif "memory_space" in self.operation.attributes:
      del self.operation.attributes["memory_space"]

  @memory_space.deleter
  def memory_space(self):
    del self.operation.attributes["memory_space"]

  @builtins.property
  def result(self):
    return self.operation.results[0]

def alloc_tensor(result, dynamic_sizes, *, copy=None, size_hint=None, memory_space=None, loc=None, ip=None) -> _ods_ir.Value:
  return AllocTensorOp(result=result, dynamic_sizes=dynamic_sizes, copy=copy, size_hint=size_hint, memory_space=memory_space, loc=loc, ip=ip).result

@_ods_cext.register_operation(_Dialect)
class CloneOp(_ods_ir.OpView):
  OPERATION_NAME = "bufferization.clone"

  _ODS_REGIONS = (0, True)

  def __init__(self, output, input, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(input)
    _ods_context = _ods_get_default_loc_context(loc)
    results.append(output)
    _ods_successors = None
    super().__init__(self.OPERATION_NAME, self._ODS_REGIONS, self._ODS_OPERAND_SEGMENTS, self._ODS_RESULT_SEGMENTS, attributes=attributes, results=results, operands=operands, successors=_ods_successors, regions=regions, loc=loc, ip=ip)

  @builtins.property
  def input(self):
    return self.operation.operands[0]

  @builtins.property
  def output(self):
    return self.operation.results[0]

def clone(output, input, *, loc=None, ip=None) -> _ods_ir.Value:
  return CloneOp(output=output, input=input, loc=loc, ip=ip).result

@_ods_cext.register_operation(_Dialect)
class DeallocOp(_ods_ir.OpView):
  OPERATION_NAME = "bufferization.dealloc"

  _ODS_OPERAND_SEGMENTS = [-1,-1,-1,]

  _ODS_REGIONS = (0, True)

  def __init__(self, memrefs, conditions, retained, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_results_or_values(memrefs))
    operands.append(_get_op_results_or_values(conditions))
    operands.append(_get_op_results_or_values(retained))
    _ods_context = _ods_get_default_loc_context(loc)
    _ods_successors = None
    super().__init__(self.OPERATION_NAME, self._ODS_REGIONS, self._ODS_OPERAND_SEGMENTS, self._ODS_RESULT_SEGMENTS, attributes=attributes, operands=operands, successors=_ods_successors, regions=regions, loc=loc, ip=ip)

  @builtins.property
  def memrefs(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operandSegmentSizes"], 0)
    return operand_range

  @builtins.property
  def conditions(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operandSegmentSizes"], 1)
    return operand_range

  @builtins.property
  def retained(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operandSegmentSizes"], 2)
    return operand_range

  @builtins.property
  def updatedConditions(self):
    _ods_variadic_group_length = len(self.operation.results) - 1 + 1
    return self.operation.results[0:0 + _ods_variadic_group_length]

def dealloc(memrefs, conditions, retained, *, loc=None, ip=None) -> _ods_ir.Value:
  return _get_op_result_or_op_results(DeallocOp(memrefs=memrefs, conditions=conditions, retained=retained, loc=loc, ip=ip))

@_ods_cext.register_operation(_Dialect)
class DeallocTensorOp(_ods_ir.OpView):
  OPERATION_NAME = "bufferization.dealloc_tensor"

  _ODS_REGIONS = (0, True)

  def __init__(self, tensor, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(tensor)
    _ods_context = _ods_get_default_loc_context(loc)
    _ods_successors = None
    super().__init__(self.OPERATION_NAME, self._ODS_REGIONS, self._ODS_OPERAND_SEGMENTS, self._ODS_RESULT_SEGMENTS, attributes=attributes, results=results, operands=operands, successors=_ods_successors, regions=regions, loc=loc, ip=ip)

  @builtins.property
  def tensor(self):
    return self.operation.operands[0]

def dealloc_tensor(tensor, *, loc=None, ip=None) -> _ods_ir.Operation:
  return DeallocTensorOp(tensor=tensor, loc=loc, ip=ip)

@_ods_cext.register_operation(_Dialect)
class MaterializeInDestinationOp(_ods_ir.OpView):
  OPERATION_NAME = "bufferization.materialize_in_destination"

  _ODS_REGIONS = (0, True)

  def __init__(self, result, source, dest, *, restrict=None, writable=None, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(source)
    operands.append(dest)
    _ods_context = _ods_get_default_loc_context(loc)
    if bool(restrict): attributes["restrict"] = _ods_ir.UnitAttr.get(
      _ods_get_default_loc_context(loc))
    if bool(writable): attributes["writable"] = _ods_ir.UnitAttr.get(
      _ods_get_default_loc_context(loc))
    if result is not None: results.append(result)
    _ods_successors = None
    super().__init__(self.OPERATION_NAME, self._ODS_REGIONS, self._ODS_OPERAND_SEGMENTS, self._ODS_RESULT_SEGMENTS, attributes=attributes, results=results, operands=operands, successors=_ods_successors, regions=regions, loc=loc, ip=ip)

  @builtins.property
  def source(self):
    return self.operation.operands[0]

  @builtins.property
  def dest(self):
    return self.operation.operands[1]

  @builtins.property
  def restrict(self):
    return "restrict" in self.operation.attributes

  @restrict.setter
  def restrict(self, value):
    if bool(value):
      self.operation.attributes["restrict"] = _ods_ir.UnitAttr.get()
    elif "restrict" in self.operation.attributes:
      del self.operation.attributes["restrict"]

  @restrict.deleter
  def restrict(self):
    del self.operation.attributes["restrict"]

  @builtins.property
  def writable(self):
    return "writable" in self.operation.attributes

  @writable.setter
  def writable(self, value):
    if bool(value):
      self.operation.attributes["writable"] = _ods_ir.UnitAttr.get()
    elif "writable" in self.operation.attributes:
      del self.operation.attributes["writable"]

  @writable.deleter
  def writable(self):
    del self.operation.attributes["writable"]

  @builtins.property
  def result(self):
    return None if len(self.operation.results) < 1 else self.operation.results[0]

def materialize_in_destination(result, source, dest, *, restrict=None, writable=None, loc=None, ip=None) -> _ods_ir.Value:
  return _get_op_result_or_op_results(MaterializeInDestinationOp(result=result, source=source, dest=dest, restrict=restrict, writable=writable, loc=loc, ip=ip))

@_ods_cext.register_operation(_Dialect)
class ToMemrefOp(_ods_ir.OpView):
  OPERATION_NAME = "bufferization.to_memref"

  _ODS_REGIONS = (0, True)

  def __init__(self, memref, tensor, *, read_only=None, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(tensor)
    _ods_context = _ods_get_default_loc_context(loc)
    if bool(read_only): attributes["read_only"] = _ods_ir.UnitAttr.get(
      _ods_get_default_loc_context(loc))
    results.append(memref)
    _ods_successors = None
    super().__init__(self.OPERATION_NAME, self._ODS_REGIONS, self._ODS_OPERAND_SEGMENTS, self._ODS_RESULT_SEGMENTS, attributes=attributes, results=results, operands=operands, successors=_ods_successors, regions=regions, loc=loc, ip=ip)

  @builtins.property
  def tensor(self):
    return self.operation.operands[0]

  @builtins.property
  def read_only(self):
    return "read_only" in self.operation.attributes

  @read_only.setter
  def read_only(self, value):
    if bool(value):
      self.operation.attributes["read_only"] = _ods_ir.UnitAttr.get()
    elif "read_only" in self.operation.attributes:
      del self.operation.attributes["read_only"]

  @read_only.deleter
  def read_only(self):
    del self.operation.attributes["read_only"]

  @builtins.property
  def memref(self):
    return self.operation.results[0]

def to_memref(memref, tensor, *, read_only=None, loc=None, ip=None) -> _ods_ir.Value:
  return ToMemrefOp(memref=memref, tensor=tensor, read_only=read_only, loc=loc, ip=ip).result

@_ods_cext.register_operation(_Dialect)
class ToTensorOp(_ods_ir.OpView):
  OPERATION_NAME = "bufferization.to_tensor"

  _ODS_REGIONS = (0, True)

  def __init__(self, result, memref, *, restrict=None, writable=None, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(memref)
    _ods_context = _ods_get_default_loc_context(loc)
    if bool(restrict): attributes["restrict"] = _ods_ir.UnitAttr.get(
      _ods_get_default_loc_context(loc))
    if bool(writable): attributes["writable"] = _ods_ir.UnitAttr.get(
      _ods_get_default_loc_context(loc))
    results.append(result)
    _ods_successors = None
    super().__init__(self.OPERATION_NAME, self._ODS_REGIONS, self._ODS_OPERAND_SEGMENTS, self._ODS_RESULT_SEGMENTS, attributes=attributes, results=results, operands=operands, successors=_ods_successors, regions=regions, loc=loc, ip=ip)

  @builtins.property
  def memref(self):
    return self.operation.operands[0]

  @builtins.property
  def restrict(self):
    return "restrict" in self.operation.attributes

  @restrict.setter
  def restrict(self, value):
    if bool(value):
      self.operation.attributes["restrict"] = _ods_ir.UnitAttr.get()
    elif "restrict" in self.operation.attributes:
      del self.operation.attributes["restrict"]

  @restrict.deleter
  def restrict(self):
    del self.operation.attributes["restrict"]

  @builtins.property
  def writable(self):
    return "writable" in self.operation.attributes

  @writable.setter
  def writable(self, value):
    if bool(value):
      self.operation.attributes["writable"] = _ods_ir.UnitAttr.get()
    elif "writable" in self.operation.attributes:
      del self.operation.attributes["writable"]

  @writable.deleter
  def writable(self):
    del self.operation.attributes["writable"]

  @builtins.property
  def result(self):
    return self.operation.results[0]

def to_tensor(result, memref, *, restrict=None, writable=None, loc=None, ip=None) -> _ods_ir.Value:
  return ToTensorOp(result=result, memref=memref, restrict=restrict, writable=writable, loc=loc, ip=ip).result
