
# # Tutorial 4: solving optimal power flow with MadNLP
#
# The previous tutorial was the last step we needed to implement
# the full optimal power flow problem with MadNLP. It just remains to add
# the following elements to the model:
#
# - the cost to run each power generator;
# - the line flow constraints;
# - the voltage angle constraints;

# We start the tutorial again by importing the usual packages:
using LinearAlgebra
using SparseArrays

using NLPModels
using ExaModels

using JLD2

include("utils.jl")

# We import a small instance:
DATA_DIR = joinpath(splitdir(Base.active_project())[1], "instances")
data = JLD2.load(joinpath(DATA_DIR, "case9.jld2"))["data"]


# ## Optimal power flow model

# We implement in ExaModels the AC optimal power flow problem in the function `acopf_model`:

function acopf_model(
    data;
    backend = nothing,
    T = Float64,
    kwargs...,
)
    w = ExaCore(T; backend = backend)
    va = variable(w, length(data.bus))
    vm = variable(
        w,
        length(data.bus);
        start = data.vm0,
        lvar = data.vmin,
        uvar = data.vmax,
    )
    pg = variable(w, length(data.gen); start=data.pg0, lvar = data.pmin, uvar = data.pmax)
    qg = variable(w, length(data.gen); start=data.qg0, lvar = data.qmin, uvar = data.qmax)
    p = variable(w, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)
    q = variable(w, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)

    o = objective(
        w,
        g.cost1 * pg[g.i]^2 + g.cost2 * pg[g.i] + g.cost3 for g in data.gen
    )

    c1 = constraint(w, va[i] for i in data.ref_buses)

    ## Active power flow, FR
    c2 = constraint(
        w,
        p[b.f_idx] - b.c5 * vm[b.f_bus]^2 -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
        b.c4 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])) for
        b in data.branch
    )
    ## Reactive power flow, FR
    c3 = constraint(
        w,
        q[b.f_idx] +
        b.c6 * vm[b.f_bus]^2 +
        b.c4 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])) for
        b in data.branch
    )
    ## Active power flow, TO
    c4 = constraint(
        w,
        p[b.t_idx] - b.c7 * vm[b.t_bus]^2 -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
        b.c2 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])) for
        b in data.branch
    )
    ## Reactive power flow, TO
    c5 = constraint(
        w,
        q[b.t_idx] +
        b.c8 * vm[b.t_bus]^2 +
        b.c2 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])) for
        b in data.branch
    )

    ## Voltage angle difference
    c6 = constraint(
        w,
        va[b.f_bus] - va[b.t_bus] for b in data.branch;
        lcon = data.angmin,
        ucon = data.angmax,
    )
    ## Line flow constraints
    c7 = constraint(
        w,
        p[b.f_idx]^2 + q[b.f_idx]^2 - b.rate_a_sq for b in data.branch;
        lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf),
    )
    c8 = constraint(
        w,
        p[b.t_idx]^2 + q[b.t_idx]^2 - b.rate_a_sq for b in data.branch;
        lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf),
    )

    ## Active power balance
    c9 = constraint(w, b.pd + b.gs * vm[b.i]^2 for b in data.bus)
    c11 = constraint!(w, c9, a.bus => p[a.i] for a in data.arc)
    c13 = constraint!(w, c9, g.bus => -pg[g.i] for g in data.gen)
    ## Reactive power balance
    c10 = constraint(w, b.qd - b.bs * vm[b.i]^2 for b in data.bus)
    c12 = constraint!(w, c10, a.bus => q[a.i] for a in data.arc)
    c14 = constraint!(w, c10, g.bus => -qg[g.i] for g in data.gen)

    return ExaModel(w; kwargs...)
end

# Solving `case9` is straightforward using MadNLP:
using MadNLP

nlp = acopf_model(data)
results = madnlp(nlp)
nothing

# ## Solving optimal power flow on the GPU
# For solving the optimal power flow model on the GPU, the set-up is similar to
# what we have detailed in the tutorial 3. We start by importing MadNLPGPU, and we
# instantiate a new optimal power flow instance on the GPU:
using CUDA
using MadNLPGPU

nlp_gpu = acopf_model(data; backend=CUDABackend())

# Solving the problem using cuDSS simply amounts to
results = madnlp(nlp_gpu)
nothing

# The instance `case9` is too small to get any significant speed-up compared
# to the CPU. However, we can solve a larger instance just by importing new data.
# For instance, to solve the case `10000_goc`:
data = JLD2.load(joinpath(DATA_DIR, "pglib_opf_case10000_goc.jld2"))["data"]
nlp_gpu = acopf_model(data; backend=CUDABackend())
results = madnlp(nlp_gpu)
nothing

