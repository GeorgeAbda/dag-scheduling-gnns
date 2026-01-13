# Cogito Library (Private)

This directory contains the proprietary `cogito` library which is not publicly available due to copyright restrictions.

## For Reviewers

If you are a reviewer and need access to run the code, please contact the authors for access credentials.

## Structure

The `cogito` library provides:
- `cogito.gnn_deeprl_model` - Deep RL models with GNN architectures for scheduling
- `cogito.dataset_generator` - DAG dataset generation and synthetic workload creation
- `cogito.tools` - Utility tools for evaluation and analysis
- `cogito.viz_results` - Visualization and result analysis tools
- `cogito.config` - Configuration management

## Installation

This library is included as a private dependency. The Python source code is protected and not included in the public repository.

## Note

All imports in `release_new/` and `scripts/` have been updated to use `cogito` instead of `scheduler`, and `gnn_deeprl_model` instead of `rl_model`.
