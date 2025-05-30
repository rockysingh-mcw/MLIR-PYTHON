import ctypes
import logging
import os
import platform
import warnings
from pathlib import Path
from typing import Union

import numpy as np

from ... import _mlir_libs
from ...dialects.func import CallOp, FuncOp
from ...ir import InsertionPoint, MemRefType, Module, UnitAttr

try:
    from ...execution_engine import ExecutionEngine
    from ...runtime import (
        UnrankedMemRefDescriptor,
        get_ranked_memref_descriptor,
        unranked_memref_to_numpy,
        get_unranked_memref_descriptor,
    )
except ImportError:
    pass

from .. import types as T
from ...dialects.memref import cast
from ..runtime.passes import Pipeline, run_pipeline
from ..util import (
    memref_type_to_np_dtype,
    mlir_type_to_ctype,
    np_dtype_to_mlir_type,
)
from ..util import shlib_ext, find_ops, shlib_prefix

logger = logging.getLogger(__name__)

# adapted from https://github.com/llvm/torch-mlir/blob/main/python/torch_mlir_e2e_test/linalg_on_tensors_backends/refbackend.py

"""
These are the expected names of the shared libraries that are used by the refbackend.
They are used to load the async runtime and the c runner utils.
If the environment variables ASYNC_RUNTIME_LIB_PATH and C_RUNNER_UTILS_LIB_PATH are set, they will be used instead.
If they are not set, then we will set the paths to the shared libraries in the same directory as this file.
The shared libraries are expected to be named mlir_async_runtime.so and mlir_c_runner_utils.so.
If you want to use the refbackend with a different set of shared libraries, you can set the environment variables ASYNC_RUNTIME_LIB_PATH and C_RUNNER_UTILS_LIB_PATH to the paths of the shared libraries.


These are compiled code of mlir

I am able to understand it but i want some more clarity on this.

why we are using the shared libraries...?
"""
CONSUME_RETURN_CALLBACK_ATTR = "refbackend_consume_return_callback"
refback_cb_attr = CONSUME_RETURN_CALLBACK_ATTR

if ASYNC_RUNTIME_LIB_PATH := os.getenv("ASYNC_RUNTIME_LIB_PATH"):
    ASYNC_RUNTIME_LIB_PATH = Path(ASYNC_RUNTIME_LIB_PATH)
else:
    ASYNC_RUNTIME_LIB_PATH = (
        Path(_mlir_libs.__file__).parent
        / f"{shlib_prefix()}mlir_async_runtime.{shlib_ext()}"
    )

if C_RUNNER_UTILS_LIB_PATH := os.getenv("C_RUNNER_UTILS_LIB_PATH"):
    C_RUNNER_UTILS_LIB_PATH = Path(C_RUNNER_UTILS_LIB_PATH)
else:
    C_RUNNER_UTILS_LIB_PATH = (
        Path(_mlir_libs.__file__).parent
        / f"{shlib_prefix()}mlir_c_runner_utils.{shlib_ext()}"
    )

if RUNNER_UTILS_LIB_PATH := os.getenv("RUNNER_UTILS_LIB_PATH"):
    RUNNER_UTILS_LIB_PATH = Path(RUNNER_UTILS_LIB_PATH)
else:
    RUNNER_UTILS_LIB_PATH = (
        Path(_mlir_libs.__file__).parent
        / f"{shlib_prefix()}mlir_runner_utils.{shlib_ext()}"
    )


# we are extracting correct ctype for mlir return types
def get_ctype_func(mlir_ret_types):
    ctypes_arg = [None]
    legal_ret_types = []
    for type in mlir_ret_types:
        if ctype := mlir_type_to_ctype(type):
            ctypes_arg.append(ctype)
            legal_ret_types.append(type)
        elif memref_type_to_np_dtype(type):
            ctypes_arg.append(ctypes.POINTER(UnrankedMemRefDescriptor))
            legal_ret_types.append(type)
        else:
            raise ValueError(f"Not supported type for callback return: {type=}")

    return ctypes.CFUNCTYPE(*ctypes_arg), legal_ret_types


""""
So at the end we have a legal return type list and a ctypes function type
"""

# https://stackoverflow.com/a/68198336/9045206
CData = ctypes._SimpleCData.__mro__[-2]

"""
CData = ct._SimpleCData.__mro__[-2]
CData, type(CData)
(<class '_ctypes._CData'>, <class 'type'>) 
In this way we are able to check if the arg is a ctypes object or not
"""


# This is used to convert the arguments from numpy arrays to ctype objects
def convert_arg_to_ctype(arg, unranked=True):
    """
    Checking here if arg is a ctype object or arg is a int float or bool, if it is, it will directly return the arg
     if it is a numpy array, we will check if the dtype is supported and return the pointer to the memref descriptor
    if it is not a numpy array, we will raise an error
    """
    if isinstance(arg, CData) or isinstance(arg, (int, float, bool)):
        return arg
    elif isinstance(arg, np.ndarray):
        assert np_dtype_to_mlir_type(
            arg.dtype.type
        ), f"unsupported numpy array type {arg.dtype}"
        if unranked:
            return ctypes.pointer(ctypes.pointer(get_unranked_memref_descriptor(arg)))
        else:
            return ctypes.pointer(
                # TODO(max): sometimes these need to be unranked memref descriptors
                ctypes.pointer(get_ranked_memref_descriptor(arg))
            )
    raise ValueError(f"unsupported {arg=} for conversion to ctype")


"""
args = [ctypes.c_float(3.14), memref_ptr]
mlir_types = [T.f32, MemRefType("f32", [2, 2])]
output from below function will be:
(
    ctypes.c_float(3.14),
    np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # From MemRef the data in the array will depend upon memref_ptr
)
"""

""" 
This is used to convert the return values from ctype to numpy arrays

"""


def convert_returns_from_ctype(args, mlir_types):
    return tuple(
        (
            arg
            if mlir_type_to_ctype(type)
            else unranked_memref_to_numpy(arg, memref_type_to_np_dtype(type))
        )
        for arg, type in zip(args, mlir_types)
    )


class LLVMJITBackendInvoker:
    def __init__(
        self,
        module,
        opt_level=2,
        shared_lib_paths=None,
        return_func_types=None,
        return_func_name=None,
        consume_return_callback=None,
    ):
        if shared_lib_paths is None:
            shared_lib_paths = []
        self.ee = ExecutionEngine(
            module, opt_level=opt_level, shared_libs=shared_lib_paths
        )
        self.results = None
        if return_func_types is not None:
            assert (
                return_func_name is not None
            ), f"must provide return func name when providing return func types"
            ctype_wrapper, ret_types = get_ctype_func(return_func_types)
            self.ret_types = ret_types
            if consume_return_callback is None:

                def consume_return_callback(*args):
                    self.results = convert_returns_from_ctype(args, self.ret_types)

            self.ee.register_runtime(
                return_func_name,
                ctype_wrapper(consume_return_callback),
            )

    def __getattr__(self, function_name: str):
        def invoke(*args):
            self.ee.invoke(
                function_name, *[convert_arg_to_ctype(a, unranked=False) for a in args]
            )
            if self.results is not None and len(self.results) == 1:
                return self.results[0]
            return self.results

        return invoke


""" 
A return consumer is a trampoline to a python function that will store/capture the return from the return;
 this is done because you can't return structs etc from C APIs.
"""


def make_return_consumer(kernel_func):
    print("\n|||------------------make return consumer ------------------------|||\n")
    c_api_compatible_types = [
        T.memref(element_type=t.element_type) if MemRefType.isinstance(t) else t
        for t in kernel_func.function_type.value.results
    ]
    cb = FuncOp(
        f"{kernel_func.name.value}_return_consumer",
        (c_api_compatible_types, []),
        visibility="private",
    )
    cb.attributes["llvm.emit_c_interface"] = UnitAttr.get()
    cb.attributes[refback_cb_attr] = UnitAttr.get()
    return cb


# A kernel wrapper is the c api interface that can be called from python (or anywhere else that C FFI is possible).
# This function will be called KERNEL_NAME_capi_wrapper and will have a {llvm.emit_c_interface} attribute.
# Note there might be other such functions in the final module (gpu-lower-to-nvvm-pipeline somehow also inserts some like this).
def make_kernel_wrapper(kernel_func, return_consumer=None):
    print("\n|||------------------make kernel wrapper ------------------------|||\n")
    input_types = kernel_func.function_type.value.inputs

    @FuncOp.from_py_func(*input_types, name=f"{kernel_func.name.value}_capi_wrapper")
    def wrapper(*args, **_kwargs):
        results = CallOp(kernel_func, list(args)).results
        if return_consumer is not None:
            c_api_compatible_results = []
            for i, a in enumerate(results):
                if MemRefType.isinstance(a.type):
                    a = cast(T.memref(element_type=a.type.element_type), a)
                c_api_compatible_results.append(a)
            CallOp(return_consumer, c_api_compatible_results)

    wrapper_func_op = wrapper.func_op
    wrapper_func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()


class LLVMJITBackend:
    def __init__(
        self,
        shared_lib_paths=None,
    ):
        if shared_lib_paths is None:
            shared_lib_paths = []
        if platform.system() != "Windows":
            shared_lib_paths += [
                ASYNC_RUNTIME_LIB_PATH,
                C_RUNNER_UTILS_LIB_PATH,
                RUNNER_UTILS_LIB_PATH,
            ]
        self.shared_lib_paths = shared_lib_paths
        self.return_func_types = None
        self.return_func_name = None

    def generate_c_api(
        self,
        module,
        kernel_name="main",
        generate_kernel_wrapper=True,
        generate_return_consumer=True,
        return_consumer=None,
    ):
        assert (
            bool(return_consumer) != bool(generate_return_consumer)
            or bool(return_consumer) == bool(generate_return_consumer) == False
        ), f"{return_consumer=} XOR {generate_return_consumer=} (or neither)"

        def cb(op):
            try:
                return kernel_name == op.sym_name.value
            except:
                return False

        kernel_func = find_ops(module.operation, cb, single=True)
        if isinstance(kernel_func, list) and len(kernel_func) == 0:
            raise ValueError(f"couldn't find kernel_func {kernel_name=}")
        if len(kernel_func.function_type.value.results) and generate_return_consumer:
            with InsertionPoint(module.body):
                return_consumer = make_return_consumer(kernel_func)

        kernel_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        if generate_kernel_wrapper:
            with InsertionPoint(module.body):
                make_kernel_wrapper(kernel_func, return_consumer)

        if return_consumer:
            self.return_func_name = return_consumer.attributes["sym_name"].value
            # this is confusing, but it's because the callback takes as operands the return values it's going to consume
            self.return_func_types = [
                i for i in return_consumer.attributes["function_type"].value.inputs
            ]

    def compile(
        self,
        module: Module,
        pipeline: Union[str, Pipeline],
        kernel_name="main",
        enable_ir_printing=False,
        generate_kernel_wrapper=True,
        generate_return_consumer=True,
        return_consumer=None,
        verify=True,
    ):
        print(
            "\n|||------------------ compilation started ------------------------|||\n"
        )
        assert (
            bool(return_consumer) != bool(generate_return_consumer)
            or bool(return_consumer) == bool(generate_return_consumer) == False
        ), f"{return_consumer=} XOR {generate_return_consumer=} (or neither)"

        pipeline = str(pipeline)
        if "to-llvm" in pipeline or generate_kernel_wrapper:
            self.generate_c_api(
                module,
                kernel_name,
                generate_kernel_wrapper,
                generate_return_consumer,
                return_consumer,
            )

        return run_pipeline(
            module,
            pipeline=pipeline,
            description="Lowering IR",
            enable_ir_printing=enable_ir_printing,
            verify=verify,
        )

    # python: /project/llvm-project/llvm/lib/Transforms/Vectorize/SLPVectorizer.cpp:821:
    # llvm::Instruction* {anonymous}::InstructionsState::getMainOp() const: Assertion `valid() && "InstructionsState is invalid."' failed.
    def load(
        self, module, consume_return_callback=None, opt_level=0
    ) -> LLVMJITBackendInvoker:
        return LLVMJITBackendInvoker(
            module,
            opt_level=opt_level,
            shared_lib_paths=[str(p.absolute()) for p in self.shared_lib_paths],
            return_func_types=self.return_func_types,
            return_func_name=self.return_func_name,
            consume_return_callback=consume_return_callback,
        )
