
# # Tutorial 3: solving constrained power flow with MadNLP
#
# In this third tutorial, we look at a variant of the power flow equations,
# where we incorporate operational constraints on the different variables: we add
# bounds on the voltage magnitude, the active and the reactive power genenerations.
# Our goal is to identify if a solution of the power flow equations exists within these bounds
# (without implementing a proper PV/PQ switching routine as in matpower).

# We start by importing the usual packages:
using LinearAlgebra
using SparseArrays

using NLPModels
using ExaModels

using JLD2

include("utils.jl")

# We import a small instance:
DATA_DIR = joinpath(splitdir(Base.active_project())[1], "instances")
data = JLD2.load(joinpath(DATA_DIR, "case9.jld2"))["data"]
ngen = length(data.gen)
nbus = length(data.bus)
nlines = length(data.branch);

# ## Constrained power flow
# On the contrary to the Tutorial 2, we consider again the power flow equations with a batch size equal to 1.
# The bounds are easy to define in ExaModels, as we can pass them to the model directly when calling the function `variable` using the keywords `lvar` and `uvar`. We use the bounds specified in the
# data. As a results, the variables are initialized as follows:
core = ExaCore()

va = variable(core, nbus)
vm = variable(core, nbus; start = data.vm0, lvar = data.vmin, uvar = data.vmax)
pg = variable(core, ngen;  start=data.pg0, lvar = data.pmin, uvar = data.pmax)
qg = variable(core, ngen;  start=data.qg0, lvar = data.qmin, uvar = data.qmax)
p = variable(core, 2*nlines)
q = variable(core, 2*nlines);

# As we obtain a bounded feasible set, we are not guaranteed to find a solution
# of the power flow constraints satisfying also the bound constraints. As a result, we
# relax the power flow constraints and penalize their violation in the objective using a ℓ1 penalty.
# If we denote by ``g(x) = 0`` the original power flow equations, the relaxed model writes
# ```math
# g(x) = σ_p - σ_n  \; , \; σ_p ≥ 0 \; , \; σ_n ≥ 0
# ```
# and we define the penalization in the objective as ``f(σ) = 1^⊤ σ_P + 1^⊤ σ_N``.

# The variables ``σ`` and the objective are defined in ExaModels as
spp = variable(core, nbus; lvar=0.0)
spn = variable(core, nbus; lvar=0.0)
sqp = variable(core, nbus; lvar=0.0)
sqn = variable(core, nbus; lvar=0.0)

objective = objective(
    core,
    spp[b.i] + spn[b.i] + sqp[b.i] + sqn[b.i] for b in data.bus
)


# We implement the full power flow model with bounds in the following function:

function constrained_power_flow_model(
    data;
    backend = nothing,
    T = Float64,
    kwargs...
)
    ngen = length(data.gen)
    nbus = length(data.bus)
    nlines = length(data.branch)

    pv_buses = get_pv_buses(data)
    free_gen = get_free_generators(data)

    w = ExaCore(T; backend = backend)

    va = variable(w, nbus)
    vm = variable(
        w,
        nbus;
        start = data.vm0,
        lvar = data.vmin,
        uvar = data.vmax,
    )
    pg = variable(w, ngen;  start=data.pg0, lvar = data.pmin, uvar = data.pmax)
    qg = variable(w, ngen;  start=data.qg0, lvar = data.qmin, uvar = data.qmax)
    p = variable(w, 2*nlines)
    q = variable(w, 2*nlines)
    ## slack variables
    spp = variable(w, nbus; lvar=0.0)
    spn = variable(w, nbus; lvar=0.0)
    sqp = variable(w, nbus; lvar=0.0)
    sqn = variable(w, nbus; lvar=0.0)

    ## Fix variables to setpoint
    c1 = constraint(w, va[i] for i in data.ref_buses)
    c01 = constraint(w, vm[i] for i in pv_buses; lcon=data.vm0[pv_buses], ucon=data.vm0[pv_buses])
    c02 = constraint(w, pg[i] for i in free_gen; lcon=data.pg0[free_gen], ucon=data.pg0[free_gen])

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

    ## Power flow constraints
    c9 = constraint(w, b.pd + b.gs * vm[b.i]^2 - spp[b.i] + spn[b.i] for b in data.bus)
    c10 = constraint(w, b.qd - b.bs * vm[b.i]^2 - sqp[b.i] + sqn[b.i] for b in data.bus)
    c11 = constraint!(w, c9, a.bus => p[a.i] for a in data.arc)
    c12 = constraint!(w, c10, a.bus => q[a.i] for a in data.arc)
    c13 = constraint!(w, c9, g.bus => -pg[g.i] for g in data.gen)
    c14 = constraint!(w, c10, g.bus => -qg[g.i] for g in data.gen)

    o = objective(
        w,
        spp[b.i] + spn[b.i] + sqp[b.i] + sqn[b.i] for b in data.bus
    )
    return ExaModel(w; kwargs...)
end

# ## Solution with the interior-point solver MadNLP
#
# We generate a new model using our function `constrained_power_flow_model`:

nlp = constrained_power_flow_model(data)
nothing

# As we have incorporated bounds on our optimization variables, the constrained power flow
# is not solvable using the Newton method we used in the two previous tutorials. However,
# it is good candidate for an interior-point method, as implemented in MadNLP.

# MadNLP takes as input any model following the `AbstractNLPModel` abstraction, as it is
# the case with our model `nlp`. As a consequence, solving the constrained power flow equations simply amounts to
# call the function `madnlp`:

using MadNLP
results = madnlp(nlp)
nothing

# We observe that MadNLP converges with a final objective close to 0, meaning that
# the power flow is feasible within the bounds. The solution returned by MadNLP is the
# same as those returned previously in Tutorial 1 by our custom Newton solver:
vm = results.solution[nbus+1:2*nbus]


# Observe that this is not the case on most instances. E.g., MadNLP converges to
# a solution with a nonzero objective on `89pegase`, meaning this instance does not have a solution
# of the power flow equations within bounded feasibility set.
data = JLD2.load(joinpath(DATA_DIR, "pglib_opf_case89_pegase.jld2"))["data"]
nlp = constrained_power_flow_model(data)
results = madnlp(nlp)
nothing

# ## Deporting the solution on the GPU
# Like our previous Newton algorithm, MadNLP supports offloading the solution of the
# model on the GPU using the extension MadNLPGPU:
using CUDA
using MadNLPGPU

# Once MadNLPGPU is imported, you just have to instantiate the previous model on the GPU to solve it using the
# same `madnlp` function:
nlp_gpu = constrained_power_flow_model(data; backend=CUDABackend())
results = madnlp(nlp_gpu)
nothing

# MadNLP detects automatically that the ExaModel instance `nlp_gpu` has been instantiated on the GPU.
# As a result the solver is able to solve the instance entirely on the GPU with the linear solver cuDSS.
# Note that we converge to the same objective value, but the number of iterations is different,
# as well as the final convergence tolerance (`tol=1e-4`): when solving a model on the GPU with cuDSS, MadNLP has to use
# a few numerical tricks that impact slightly the accuracy in the evaluation. The tolerance has to be loosened to obtain a reliable convergence on the GPU. If you find the solution not satisfactory, you can specify
# your own convergence tolerance by using the option `tol`. E.g., to solve the model with the same precision as on the CPU:
results = madnlp(nlp_gpu; tol=1e-8)
nothing

# We have now all the elements in hand to solve the full optimal power flow problem on the GPU using MadNLP.

