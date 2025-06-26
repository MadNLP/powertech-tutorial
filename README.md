# ExaModels Tutorial

This repository contains the script to generate the ExaModels tutorial documentation and the Jupyter notebooks.

# How to run
First, obtain the instances file via 
```bash
wget https://cloud.minesparis.psl.eu/index.php/s/8nfxDqzz41H0rpE/download
unzip download
```
Next, instantiate the Julia environment by running:
```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```
Then, you can run the script to generate the documentation and Jupyter notebooks:
```julia
include("make.jl")
include("docs/make.jl")
```
