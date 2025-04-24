import numpy as np

import mlir.extras.types as T
from mlir.dialects import builtin
from mlir.dialects.transform import any_op_t
from mlir.dialects.transform.extras import named_sequence, apply_patterns
from mlir.extras.util import find_ops
from mlir.ir import StringAttr, UnitAttr

# you need this to register the memref value caster
# noinspection PyUnresolvedReferences
import mlir.extras.dialects.ext.memref
from mlir.extras.context import RAIIMLIRContext, ExplicitlyManagedModule
from mlir.dialects.bufferization import LayoutMapOption
from mlir.dialects.transform.vector import (
    VectorContractLowering,
    VectorMultiReductionLowering,
    VectorTransferSplit,
    VectorTransposeLowering,
)
from mlir.extras.dialects.ext import linalg
from mlir.extras.dialects.ext.func import func
from mlir.extras.dialects.ext.transform import (
    match,
)
from mlir.extras.dialects.ext import transform
from mlir.extras.runtime.passes import Pipeline, run_pipeline
from mlir.extras.runtime.refbackend import LLVMJITBackend

ctx = RAIIMLIRContext()
backend = LLVMJITBackend()
module = ExplicitlyManagedModule()


@func
def add_tensors(
    A: T.tensor(3, 3, T.f32()),
    B: T.tensor(3, 3, T.f32()),
    C: T.tensor(3, 3, T.f32()),
):
    return linalg.add(A, B, C)


print(add_tensors.emit())


@builtin.module(attrs={"transform.target_tag": StringAttr.get("payload")})
def payload():
    add_tensors.emit(force=True)


@builtin.module(attrs={"transform.with_named_sequence": UnitAttr.get()})
def mod_transform():
    @named_sequence("main", [any_op_t()], [])
    def main(module_op: any_op_t()):
        add = match(module_op, ops=["linalg.add"])
        # transform.structured.vectorize(add, vector_sizes=[])


# module = module.finish()
# print(module)

vectorized_module = run_pipeline(
    module,
    pipeline=Pipeline().transform_interpreter(
        entry_point="main", debug_payload_root_tag="payload"
    ),
)

print(vectorized_module)

# https://github.com/makslevental/llvm-project/blob/f6643263631bcb0d191ef923963ac1a5ca9ac5fd/mlir/test/lib/Dialect/LLVM/TestLowerToLLVM.cpp#L44
lower_to_llvm = (
    Pipeline()
    .Func(
        Pipeline()
        # Blanket-convert any remaining high-level vector ops to loops if any remain.
        .convert_vector_to_scf()
        # Blanket-convert any remaining linalg ops to loops if any remain.
        .convert_linalg_to_loops()
    )
    # Blanket-convert any remaining affine ops if any remain.
    .lower_affine()
    # Convert SCF to CF (always needed).
    .convert_scf_to_cf()
    # Sprinkle some cleanups.
    .canonicalize()
    .cse()
    # Convert vector to LLVM (always needed).
    .convert_vector_to_llvm()
    # Convert Math to LLVM (always needed).
    .Func(Pipeline().convert_math_to_llvm())
    # Expand complicated MemRef operations before lowering them.
    .expand_strided_metadata()
    # The expansion may create affine expressions. Get rid of them.
    .lower_affine()
    # Convert MemRef to LLVM (always needed).
    .finalize_memref_to_llvm()
    # Convert Func to LLVM (always needed).
    .convert_func_to_llvm()
    .convert_arith_to_llvm()
    .convert_cf_to_llvm()
    # Convert Index to LLVM (always needed).
    .convert_index_to_llvm()
    # Convert remaining unrealized_casts (always needed).
    .reconcile_unrealized_casts()
)


# compiled_module = backend.compile(
#     find_ops(
#         vectorized_module.operation,
#         lambda x: "transform.target_tag" in x.attributes
#         and x.attributes["transform.target_tag"].value == "payload",
#         single=True,
#     ),
#     kernel_name=add_tensors.__name__,
#     pipeline=lower_to_llvm,
# )

# # print(compiled_module)

# A = np.random.randint(0, 10, (M, K)).astype(np.float32)
# B = np.random.randint(0, 10, (K, N)).astype(np.float32)
# C = np.zeros((M, N), dtype=np.float32)

# backend.load(compiled_module).matmul_tensors_capi_wrapper(A, B, C)
# assert np.allclose(A @ B, C)
