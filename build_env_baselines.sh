#!/bin/bash
set -xe

# Default values for arguments
env_name=l1ra_baseline
cuda_version=122

# Help function
show_help() {
    echo "Usage: $0 [-n <environment name>] [-h]"
    echo
    echo "Options:"
    echo "  -n    Set the name of the conda environment (default: ${env_name})"
    echo "  -h    Display this help message"
    echo
    echo "Example:"
    echo "  ${0} -n myenv"
}

# Parse command-line arguments
while getopts "n:h" opt; do
    case ${opt} in
        n )
            env_name=${OPTARG}
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
#
conda run -n ${env_name} pip install -r requirements_baseline.txt
conda run -n ${env_name} conda env config vars set PYTHONPATH=${PYTHONPATH}:${PWD}/src/

exit 0
