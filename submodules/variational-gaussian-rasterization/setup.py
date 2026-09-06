#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys

# NOTE: This is a modified version of the original setup.py.
# --- Force CUDA from the active conda env ------------------------------------
# Many build systems locate nvcc as f"{CUDA_HOME}/bin/nvcc". On your machine,
# CUDA_HOME was pointing to the *base* conda. We pin it to the active env.

conda_prefix = os.environ.get("CONDA_PREFIX")
if not conda_prefix:
    raise RuntimeError(
        "CONDA_PREFIX is not set. Activate your conda environment before building."
    )

cuda_root = conda_prefix  # conda installs cuda-toolkit under the env root
nvcc_path = os.path.join(cuda_root, "bin", "nvcc")

# Prepend env CUDA bin to PATH and advertise toolkit root via common vars
# Clear any stale pointers
for var in ("CUDA_HOME", "CUDAHOME", "CUDA_PATH"):
    os.environ.pop(var, None)

os.environ["CUDA_HOME"]  = cuda_root
os.environ["CUDAHOME"]   = cuda_root
os.environ["PATH"]       = os.pathsep.join([os.path.join(cuda_root, "bin"),
                                            os.environ.get("PATH", "")])
# Ensure libs are discoverable at link/run time
ld_paths = [os.path.join(cuda_root, "lib"), os.path.join(cuda_root, "lib64")]
existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
if existing_ld:
    ld_paths.append(existing_ld)
os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(ld_paths)

# Fail early with a clear error if nvcc isn't available in this env
if not os.path.exists(nvcc_path):
    raise FileNotFoundError(
        f"nvcc not found at {nvcc_path}.\n"
        f"- Make sure you've installed nvcc in THIS env, e.g.:\n"
        f"    conda config --set channel_priority flexible\n"
        f"    conda install -c 'nvidia/label/cuda-11.8.0' cuda-nvcc=11.8.89\n"
        f"- Or adjust CUDA_HOME to the toolkit you want to use."
    )

# Debug prints so you can see what gets picked up in pip logs
print(f"[setup.py] Using CUDA_HOME={os.environ['CUDA_HOME']}", file=sys.stderr)
print(f"[setup.py] Using nvcc at {nvcc_path}", file=sys.stderr)
print(f"[setup.py] PATH begins: {os.environ['PATH'].split(os.pathsep)[0]}", file=sys.stderr)

# -----------------------------------------------------------------------------
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
glm_include = os.path.join(this_dir, "third_party", "glm")

setup(
    name="variational_gaussian_rasterization",
    packages=["variational_gaussian_rasterization"],
    ext_modules=[
        CUDAExtension(
            name="variational_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "nvcc": [f"-I{glm_include}"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
