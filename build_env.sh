#!/bin/bash
set -xe

# Default values for arguments
env_name=l1ra
cuda_version=122

# Help function
show_help() {
    echo "Usage: $0 [-n <environment name>] [-c <cuda version>] [-h]"
    echo
    echo "Options:"
    echo "  -n    Set the name of the conda environment (default: ${env_name})"
    echo "  -c    Set the CUDA version (default: ${cuda_version})"
    echo "  -h    Display this help message"
    echo
    echo "Example:"
    echo "  ${0} -n myenv -c 118"
}

# Parse command-line arguments
while getopts "n:c:h" opt; do
    case ${opt} in
        n )
            env_name=${OPTARG}
            ;;
        c )
            cuda_version=${OPTARG}
            ;;
        h )
            show_help
            exit 0
            ;;
        \? )
            echo "Invalid option: -${OPTARG}" >&2
            show_help
            exit 1
            ;;
        : )
            echo "Option -${OPTARG} requires an argument." >&2
            show_help
            exit 1
            ;;
    esac
done

# Create and activate Anaconda environment
conda create -n ${env_name} python=3.12 -y
mkdir -p ${CONDA_PREFIX}/envs/${env_name}/local
# Get submodule sources
git submodule init
git submodule update
# Build BNB from source (https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/docs/source/installation.mdx)
cd ./submodules/bitsandbytes
## Install CUDA
conda run -n ${env_name} conda install cuda -c nvidia/label/cuda-12.2.2
conda run -n ${env_name} conda install -c conda-forge gcc=12
conda run -n ${env_name} conda install -c conda-forge cxx-compiler
conda run -n ${env_name} bash install_cuda.sh ${cuda_version} ${CONDA_PREFIX}/envs/${env_name}/local 1
conda run -n ${env_name} conda env config vars set BNB_CUDA_VERSION=${cuda_version}
conda run -n ${env_name} conda env config vars set LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/envs/${env_name}/local/cuda-${cuda_version::2}.${cuda_version:2}
## Compile
conda run -n ${env_name} pip install -r requirements-dev.txt
conda run -n ${env_name} cmake -DCOMPUTE_BACKEND=cuda -S .
conda run -n ${env_name} make
conda run -n ${env_name} pip install -e .

exit 0
