### A Pluto.jl notebook ###
# v0.20.9

using Markdown
using InteractiveUtils

# ╔═╡ 99cce976-0674-4aa1-981e-f81ad12d6a8c
using ExaModels

# ╔═╡ 69be43d6-7260-4bea-a021-09aedede16d7
using NLPModelsIpopt

# ╔═╡ d2fd33aa-30e3-4f7f-9bd6-44b99402ba12
md"""

# ExaModels: A Novel Approach to NLP Modeling and AD

In this notebook, we will:

- Introduce the primary motivation behind the development of a novel algebraic modeling and Automatic Differentiation (AD) system.
- Differentiate ExaModels from other algebraic modeling systems, such as JuMP or Pyomo/AMPL, highlighting its distinctive features.
- Emphasize the key advantages offered by our method.
- Explore the engineering behind ExaModels.
- Discuss the possibility of integrating ExaModels as an AD backend for JuMP/MOI.

"""

# ╔═╡ 4d07f3c5-00bb-4d7e-8867-f3c038f43896
md"""

You might recall MadDiff.jl, but...

## Why Was MadDiff.jl Discontinued?

- MadDiff.jl's impressive performance was primarily driven by exploiting repeated structures.
- The computational pattern was encoded within the type of the expression tree, often resulting in deeply nested type expressions.
- Subsequently, a for loop was applied over each pattern, yielding a highly efficient loop without the need for type inference.
- We perform Automatic Differentiation (AD) without using coloring. Counters are employed to keep track of the number of evaluations. In each evaluation, values are assigned as follows:
```julia
v[cnt += 1] = calculated_derivative
```
- After careful consideration, I realized that:
  - There was no necessity to create a model with a "full" expression tree, which would have an O(# terms) size when generating the expression tree.
  - Instead, we could generate an expression tree only for each pattern, resulting in an O(# patterns) size expression tree.
  - This approach also lends itself well to parallelization on GPUs.
  - Can significantly reduce the model creation time (not counting compilation time) and memory footprint.
  - One of MadDiff's limitations was its storage of intermediate computation results within the tree. This mandated the use of mutable types, which cannot be employed within GPU kernels.
"""

# ╔═╡ 549b29d8-ac43-4b83-beda-76b95e686425
md"""

## How ExaModels Has Advanced Beyond MadDiff

- The fundamental philosophy of ExaModels closely resembles that of MadDiff. Our aim remains to create a fast Automatic Differentiation (AD) system tailored for large-scale nonlinear programming (NLP). We accomplish this by leveraging the power of deeply nested types.
- We acknowledge the trade-off during compilation but reap the rewards of enhanced runtime performance.
- We introduce what is known as the SIMD (Single Instruction, Multiple Data) abstraction in NLPs. Instead of the user providing individual constraint/objective functions, ExaModels relies on the user to provide objective/constraint patterns, greatly simplifying the process. 
```math
\begin{aligned}
  \min_{x^\flat\leq x \leq x^\sharp}
  & \sum_{l\in[L]}\sum_{i\in [I_l]} f^{(l)}(x; p^{(l)}_i)\\
  \text{s.t.}\; &\left[g^{(m)}(x; q_j)\right]_{j\in [J_m]} +\sum_{n\in [N_m]}\sum_{k\in [K_n]}h^{(n)}(x; s^{(n)}_{k}) =0,\quad \forall m\in[M]
\end{aligned}
```

- Notably, the implemented expression tree is a bittype, rendering it compatible with GPUs. This compatibility enables efficient parallelization and enhances performance

"""

# ╔═╡ c539c5f0-2f0f-4af2-a7b8-312366e5428a
md"""
## ExaModels Performance Overview

- *Baseline:* ExaModels caters to more specialized problem settings, which leads to its expected superior performance compared to JuMP/AMPL.

- *Derivative Evaluation Efficiency*: ExaModels demonstrates highly efficient derivative evaluation, surpassing the speed of MadDiff and significantly outperforming JuMP/AMPL for the specific problem categories we considered. These include scalable benchmark problems, optimal control, and optimal power flow problems.

![Benchmark Performance](https://raw.githubusercontent.com/sshin23/ExaModels.jl/main/docs/src/assets/benchmark.svg)

- *For Large-scale AC-OPFs:*
  - Automatic Differentiation (AD) on Nvidia GPU (V100) achieves approximately a 500-fold speed improvement over JuMP on CPU (Intel Xeon Gold 6140), and about 10 times faster than ExaModels running on CPU.
  - With the integration of ExaModels, we've achieved a tenfold acceleration in solving AC OPF problems using GPU (ExaModels + MadNLP + cuSOLVER) compared to the conventional approach (JuMP + Ipopt + Ma27). For further details, refer to [arxiv.org/abs/2307.16830](https://arxiv.org/abs/2307.16830).
  - Note: It's worth mentioning that currently, only one solver option, MadNLP, is available. While a buffered model can be created to make it compatible with CPU, it does cause a slight slowdown.

- *Portability:*
  - ExaModels is designed to run on a variety of GPU architectures (NVIDIA, AMD, INTEL) as well as multi-threaded CPUs.

"""

# ╔═╡ f4aff33a-0a9f-47e0-a509-0d0487cb119b
md"""
## A short tutorial of ExaModels

Let's say we want to model
```math
\begin{aligned}
\min_{\{x_i\}_{i=0}^N} &\sum_{i=2}^N  100(x_{i-1}^2-x_i)^2+(x_{i-1}-1)^2\\
\text{s.t.} &  3x_{i+1}^3+2x_{i+2}-5+\sin(x_{i+1}-x_{i+2})\sin(x_{i+1}+x_{i+2})+4x_{i+1}-x_i e^{x_i-x_{i+1}}-3 = 0
\end{aligned}
```
An ExaModels script can be written as follows.
"""

# ╔═╡ 0baa1ee7-2b20-4d42-a3c6-7ff4a3a703fa
c = ExaCore()

# ╔═╡ f1c4b57d-f091-4056-bf32-59b95b65240f
md"""
This is where our optimziation model information will be progressively stored. This object is not yet an `NLPModel`, but it will essentially store all the necessary information.

Now, let's create the optimziation variables. From the problem definition, we can see that we will need $N$ scalar variables. We will choose $N=10$, and create the variable $x\in\mathbb{R}^{N}$ with the follwoing command:
"""

# ╔═╡ d6fcac43-9b06-4891-a64b-026440cc0b70
begin
    N = 10; # problem parameter
    x = variable(c, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
end

# ╔═╡ 916ccdaf-1640-40e6-943a-e43298a6f0a5
md"""
This creates the variable `x`, which we will be able to refer to when we create constraints/objective constraionts. Also, this modifies the information in the `ExaCore` object properly so that later an optimization model can be properly created with the necessary information. Observe that we have used the keyword argument `start` to specify the initial guess for the solution. The variable upper and lower bounds can be specified in a similar manner. 
"""

# ╔═╡ 0b128573-d982-4ad9-8e95-72d376a376bc
objective(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N)

# ╔═╡ a395acde-5336-499f-94e5-015efbe7564f
constraint(
    c,
    3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
    x[i]exp(x[i] - x[i+1]) - 3 for i = 1:N-2
)

# ╔═╡ 9885c6e0-2c61-41f9-ad96-5660e4be02ea
m = ExaModel(c)

# ╔═╡ 59baf20f-7e10-4168-aab6-372a924a0368
md"""
`ExaModel` is a subtype of `NLPModels.AbstractNLPModel`. So, in principle, any solver in `JuliaSmoothOptimizer` ecosystem, such as `Ipopt`, `MadNLP`, `Percival`, etc. can solve `ExaModel`. For example,
"""

# ╔═╡ 521105b5-de12-4fa5-837a-8ba85cd7e4cb
result = ipopt(m);

# ╔═╡ 7e5edff7-fe95-4b38-9134-6a08cf6e7926
md"""
More complex optimal power flow example looks like:
```julia
function ac_power_model(
    filename = "pglib_opf_case3_lmbd.m";
    backend = nothing,
    T = Float64,
    kwargs...,
)

    data = parse_ac_power_data(filename, backend)

    w = ExaModels.ExaCore(T, backend)

    va = ExaModels.variable(w, length(data.bus);)

    vm = ExaModels.variable(
        w,
        length(data.bus);
        start = fill!(similar(data.bus, Float64), 1.0),
        lvar = data.vmin,
        uvar = data.vmax,
    )
    pg = ExaModels.variable(w, length(data.gen); lvar = data.pmin, uvar = data.pmax)

    qg = ExaModels.variable(w, length(data.gen); lvar = data.qmin, uvar = data.qmax)

    p = ExaModels.variable(w, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)

    q = ExaModels.variable(w, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)

    o = ExaModels.objective(
        w,
        g.cost1 * pg[g.i]^2 + g.cost2 * pg[g.i] + g.cost3 for g in data.gen
    )

    c1 = ExaModels.constraint(w, va[i] for i in data.ref_buses)

    c2 = ExaModels.constraint(
        w,
        p[b.f_idx] - b.c5 * vm[b.f_bus]^2 -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
        b.c4 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])) for
        b in data.branch
    )

    c3 = ExaModels.constraint(
        w,
        q[b.f_idx] +
        b.c6 * vm[b.f_bus]^2 +
        b.c4 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])) for
        b in data.branch
    )

    c4 = ExaModels.constraint(
        w,
        p[b.t_idx] - b.c7 * vm[b.t_bus]^2 -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
        b.c2 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])) for
        b in data.branch
    )

    c5 = ExaModels.constraint(
        w,
        q[b.t_idx] +
        b.c8 * vm[b.t_bus]^2 +
        b.c2 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])) for
        b in data.branch
    )

    c6 = ExaModels.constraint(
        w,
        va[b.f_bus] - va[b.t_bus] for b in data.branch;
        lcon = data.angmin,
        ucon = data.angmax,
    )
    c7 = ExaModels.constraint(
        w,
        p[b.f_idx]^2 + q[b.f_idx]^2 - b.rate_a_sq for b in data.branch;
        lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf),
    )
    c8 = ExaModels.constraint(
        w,
        p[b.t_idx]^2 + q[b.t_idx]^2 - b.rate_a_sq for b in data.branch;
        lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf),
    )

    c9 = ExaModels.constraint(w, b.pd + b.gs * vm[b.i]^2 for b in data.bus)

    c10 = ExaModels.constraint(w, b.qd - b.bs * vm[b.i]^2 for b in data.bus)

    c11 = ExaModels.constraint!(w, c9, a.bus => p[a.i] for a in data.arc)
    c12 = ExaModels.constraint!(w, c10, a.bus => q[a.i] for a in data.arc)

    c13 = ExaModels.constraint!(w, c9, g.bus => -pg[g.i] for g in data.gen)
    c14 = ExaModels.constraint!(w, c10, g.bus => -qg[g.i] for g in data.gen)

    return ExaModels.ExaModel(w; kwargs...)

end
```
"""

# ╔═╡ 55aef241-fd15-4dfb-b8d7-7980d7cc0658
md"""
A few notes:
- All the data arrays should have concrete, bit-type elements. Can be a bit challenging for casual Julia users.
- `constraint!` are used to add terms to the constraints
"""

# ╔═╡ 10038161-9f4e-478a-916d-56e5f709f366
md"""
## Engineering behind ExaModels
### Building Blocks of ExaModel

We have three different types of expression trees.
- Symbolic expression tree
  - Used for recording the computation pattern
  - created while creating a model
  - It is a "Parameterized expression tree", so it does not store data but has information about the data's structure.
  - All basic functions like `+`, `*`, `^`, `exp`, etc. are extended so that we can have convenient manipulations.
```julia
struct VarSource <: AbstractNode end


struct Var{I} <: AbstractNode
    i::I
end

struct Node1{F,I} <: AbstractNode
    inner::I
end

struct Node2{F,I1,I2} <: AbstractNode
    inner1::I1
    inner2::I2
end
```
"""

# ╔═╡ 651b85f4-2b07-421f-925d-2203ec0e5795
begin
    v = ExaModels.VarSource();
    (sin(v[2]) * exp(v[1])) / v[3]
end

# ╔═╡ 55e04369-57b0-4583-95e6-7c3a030a9a58
md"""
- First-order forward pass tree
```julia
struct AdjointNode1{F,T,I} <: AbstractAdjointNode
    x::T
    y::T
    inner::I
end

struct AdjointNode2{F,T,I1,I2} <: AbstractAdjointNode
    x::T
    y1::T
    y2::T
    inner1::I1
    inner2::I2
end

```
- Second-order forward pass tree
  - These are created temporarily while evaluating the derivatives internally within the derivative evaluation functions and are not stored.
- New functions can be registered via
```julia
@register_univariate(
    Base.sqrt,
    x -> (0.5 / sqrt(x)),
    x -> ((0.5 * -(0.5 / sqrt(x))) / sqrt(x)^2)
)
```
"""

# ╔═╡ 3fee9d47-39a5-42c5-87ac-e4ded1f15d40
md"""
### What happens to the user-provided `Generator`s?
Let's say that we have this generator
"""

# ╔═╡ c8894712-1b6a-4b37-98de-16c071e9b207
data = [(index = i, a = exp(i/100)) for i=1:10]

# ╔═╡ 35c751ad-6943-4919-8301-6e3d5bb32b23
gen = (d.a * sin(x[d.index]^2) for d in data)

# ╔═╡ ca827adf-cc47-45cf-9b4b-627b4668016f
md"""
- `Generator`s have two parts:
  - `f` (a function)
  - `iter` (an array)
- First, ExaModels inspect the type of data array. Let's say that our data is constructed by
"""

# ╔═╡ a4d71dc7-77f2-40a6-a55f-b92c666e2497
md"""
Within ExaModels, we process this by looking at `eltype` of `data`.
"""

# ╔═╡ 8a113e7a-1501-4e81-8c03-e6e49803d8df
p = ExaModels.Par(eltype(gen.iter))

# ╔═╡ 6c48af93-a204-4888-93ea-434e80025aed
md"""
What is this? It is a parameterized data, which represents the structure of our data array. This is not necessarily holding the data, but it has the information on the data's structure.

Now, we can use this to create a parameterized expression tree.
"""

# ╔═╡ a939ebe8-0608-42f9-96c5-c4764e1484c7
f = gen.f(p)

# ╔═╡ 3a8bc0ae-946d-4004-a038-afcd989a3e25
md"""
### Forward pass
```julia
function gradient!(y, f, x, adj)
    @simd for k in eachindex(f.itr)
        @inbounds gradient!(y, f.f.f, x, f.itr[k], adj)
    end
    return y
end
function gradient!(y, f, x, p, adj)
    graph = f(p, AdjointNodeSource(x))
    drpass(graph, y, adj)
    return y
end
```
"""

# ╔═╡ 7ec8f015-d923-4dfd-acc5-ea1768987cd5
md"""
### Reverse pass
```julia
@inline function drpass(d::D, y, adj) where {D<:AdjointNode1}
    offset = drpass(d.inner, y, adj * d.y)
    nothing
end
@inline function drpass(d::D, y, adj) where {D<:AdjointNode2}
    offset = drpass(d.inner1, y, adj * d.y1)
    offset = drpass(d.inner2, y, adj * d.y2)
    nothing
end
@inline function drpass(d::D, y, adj) where {D<:AdjointNodeVar}
    @inbounds y[d.i] += adj
    nothing
end
```
"""

# ╔═╡ 33daf7d0-054c-463b-9399-dfc1f5e8d5b4
md"""
- Hessian computation is a bit more complicated but have the same idea.
- But things get a bit more complicated due to sparsity
### Sparsity handling
- Actually, reverse-mode AD API can be extended to perform the sparsity detection and sparsity evaluation.
- Let's take a look at the sparse gradient case.
```julia
@inline function grpass(d::D, comp, y, o1, cnt, adj) where {D<:AdjointNode1}
    cnt = grpass(d.inner, comp, y, o1, cnt, adj * d.y)
    return cnt
end
@inline function grpass(d::D, comp, y, o1, cnt, adj) where {D<:AdjointNode2}
    cnt = grpass(d.inner1, comp, y, o1, cnt, adj * d.y1)
    cnt = grpass(d.inner2, comp, y, o1, cnt, adj * d.y2)
    return cnt
end
@inline function grpass(d::D, comp, y, o1, cnt, adj) where {D<:AdjointNodeVar}
    @inbounds y[o1+comp(cnt += 1)] += adj
    return cnt
end
@inline function grpass(d::AdjointNodeVar, comp::Nothing, y, o1, cnt, adj) # despecialization
    push!(y, d.i)
    return (cnt += 1)
end
@inline function grpass(
    d::D,
    comp,
    y::V,
    o1,
    cnt,
    adj,
) where {D<:AdjointNodeVar,V<:AbstractVector{Tuple{Int,Int}}}
    ind = o1 + comp(cnt += 1)
    @inbounds y[ind] = (d.i, ind)
    return cnt
end
```
- What are the `comp`? They record the mapping from the counter to the index. This allows us to compress the sparse gradient (same idea applied for Jacobian and Hessian as well).\
- `comp` is created as part of the initial analysis of the generators.
"""

# ╔═╡ 3e444d85-c9c1-49ea-9f28-a6c65219cc52
md"""
## Extension Packages
The GPU kernels are implemented in the extension packages
```toml
name = "ExaModels"
uuid = "1037b233-b668-4ce9-9b63-f9f681f55dd2"
authors = ["Sungho Shin <sshin@anl.gov>"]
version = "0.4.2"

[deps]
NLPModels = "a4795742-8479-5a88-8948-cc11e1c8c1a6"
SolverCore = "ff4d7338-4cf1-434d-91df-b86cb86fb843"

[weakdeps]
AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[extensions]
ExaModelsAMDGPU = "AMDGPU"
ExaModelsCUDA = "CUDA"
ExaModelsKernelAbstractions = "KernelAbstractions"
ExaModelsOneAPI = "oneAPI"
ExaModelsSpecialFunctions = "SpecialFunctions"

[compat]
AMDGPU = "0.5"
CUDA = "4"
KernelAbstractions = "0.9"
NLPModels = "0.18, 0.19, 0.20"
SolverCore = "0.3"
SpecialFunctions = "2"
julia = "1.9"
oneAPI = "1"

[extras]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Downloads = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
MadNLP = "2621e9c9-9eb4-46b1-8089-e8c72242dfb6"
NLPModels = "a4795742-8479-5a88-8948-cc11e1c8c1a6"
NLPModelsIpopt = "f4238b75-b362-5c4c-b852-0801c9a21d71"
NLPModelsJuMP = "792afdf1-32c1-5681-94e0-d7bf7a5df49e"
Percival = "01435c0c-c90d-11e9-3788-63660f8fbccc"
PowerModels = "c36e90e8-916a-50a6-bd94-075b64ef4655"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test", "NLPModels", "NLPModelsIpopt", "KernelAbstractions", "CUDA", "AMDGPU", "oneAPI", "MadNLP", "Percival", "PowerModels", "JuMP", "NLPModelsJuMP", "Downloads", "Random", "ForwardDiff", "SpecialFunctions"]
```
"""

# ╔═╡ da9e5ec9-db5a-441a-b6a4-b70f80e9bdcf
md"""
## Can ExaModels be interfaced with JuMP?
- In principle, I believe yes. I have interfaced MadDiff.jl with it, and I believe there will be no reason we won't be able to do it.
- But it will be a bit more difficult than MadDiff as we use SIMD abstraction.
- InfiniteOpt.jl, for example, can have more straight-frward way of writing the interface.
- But we can inspect the expressions one by one, classify them, and write them as ExaModel. So, possible in principle, but there will be significant overhead.
- In the long-term, maybe it is a good idea for JuMP to support objective/constraint "templates" (e.g. Gravity). 
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ExaModels = "1037b233-b668-4ce9-9b63-f9f681f55dd2"
NLPModelsIpopt = "f4238b75-b362-5c4c-b852-0801c9a21d71"

[compat]
ExaModels = "~0.4.2"
NLPModelsIpopt = "~0.10.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0-beta2"
manifest_format = "2.0"
project_hash = "fc37bc5e8040ab4880717044a350ce7273c3c12d"

[[deps.AMD]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse_jll"]
git-tree-sha1 = "d4b99dd70d7136fe75ec74d072191d688448d39c"
uuid = "14f7f29c-3bd6-536c-9a0b-7339e30b5a3e"
version = "0.5.2"

[[deps.ASL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6252039f98492252f9e47c312c8ffda0e3b9e78d"
uuid = "ae81ac8f-d209-56e5-92de-9978fef736f9"
version = "0.1.3+0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "ad41de3795924f7a056243eb3e4161448f0523e6"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "02aa26a4cf76381be7f66e020a3eddeb27b0a092"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "e460f044ca8b99be31d35fe54fc33a5c33dd8ed7"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.9.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.ExaModels]]
deps = ["NLPModels", "SolverCore"]
git-tree-sha1 = "054b99d8f8d19a81a1fbbad74223af16f8012f2a"
uuid = "1037b233-b668-4ce9-9b63-f9f681f55dd2"
version = "0.4.2"

    [deps.ExaModels.extensions]
    ExaModelsAMDGPU = "AMDGPU"
    ExaModelsCUDA = "CUDA"
    ExaModelsKernelAbstractions = "KernelAbstractions"
    ExaModelsOneAPI = "oneAPI"
    ExaModelsSpecialFunctions = "SpecialFunctions"

    [deps.ExaModels.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "70379ec206165d3ca0c259d38289a6366f5398a1"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.9.2+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Ipopt]]
deps = ["Ipopt_jll", "LinearAlgebra", "MathOptInterface", "OpenBLAS32_jll", "PrecompileTools"]
git-tree-sha1 = "e2a6bf921d9569e2a07857518c7ee3afb783f554"
uuid = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
version = "1.4.2"

[[deps.Ipopt_jll]]
deps = ["ASL_jll", "Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "MUMPS_seq_jll", "SPRAL_jll", "libblastrampoline_jll"]
git-tree-sha1 = "f06a7fd68e29c8acc96483d6f163dab58626c4b5"
uuid = "9cc047cb-c261-5740-88fc-0cf96f7bdcc7"
version = "300.1400.1302+0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LDLFactorizations]]
deps = ["AMD", "LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "cbf4b646f82bfc58bb48bcca9dcce2eb88da4cd1"
uuid = "40e66cde-538c-5869-a4ad-c39174c6795b"
version = "0.10.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.0.1+1"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearOperators]]
deps = ["FastClosures", "LDLFactorizations", "LinearAlgebra", "Printf", "SparseArrays", "TimerOutputs"]
git-tree-sha1 = "a58ab1d18efa0bcf9f0868c6d387e4126dad3e72"
uuid = "5c8ed15e-5a4c-59e4-a42b-c7e8811fb125"
version = "2.5.2"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.METIS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "1fd0a97409e418b78c53fac671cf4622efdf0f21"
uuid = "d00139f3-1899-568f-a2f0-47f597d42d70"
version = "5.1.2+0"

[[deps.MUMPS_seq_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "METIS_jll", "libblastrampoline_jll"]
git-tree-sha1 = "24dd34802044008ef9a596de32d63f3c9ddb7802"
uuid = "d7ed1dd3-d0ae-5e8e-bfb4-87a502085b8d"
version = "500.600.100+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "8bfc1519e9de0564d378b3886b21b9a2f04cbdb5"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.20.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "a7b647ce8f4fefbcaf7de28fa208c812e21dc18f"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.3.2"

[[deps.NLPModels]]
deps = ["FastClosures", "LinearAlgebra", "LinearOperators", "Printf", "SparseArrays"]
git-tree-sha1 = "51b458add76a938917772ee661ffb9d59b4c7e5d"
uuid = "a4795742-8479-5a88-8948-cc11e1c8c1a6"
version = "0.20.0"

[[deps.NLPModelsIpopt]]
deps = ["Ipopt", "NLPModels", "SolverCore"]
git-tree-sha1 = "fc840dec9e9d371e3747b2c5c85c2a58ed99d124"
uuid = "f4238b75-b362-5c4c-b852-0801c9a21d71"
version = "0.10.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "98340f566166bb193a2facaffec2fbb36246802a"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.23+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+2"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SPRAL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "Libdl", "METIS_jll", "libblastrampoline_jll"]
git-tree-sha1 = "d1ca34081034a9c6903cfbe068a952a739c2aa5c"
uuid = "319450e9-13b8-58e8-aa9f-8fd1420848ab"
version = "2023.8.2+0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SolverCore]]
deps = ["LinearAlgebra", "NLPModels", "Printf"]
git-tree-sha1 = "9fb0712d597d6598857ae50b7744df17b1137b38"
uuid = "ff4d7338-4cf1-434d-91df-b86cb86fb843"
version = "0.3.7"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.0+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─d2fd33aa-30e3-4f7f-9bd6-44b99402ba12
# ╟─4d07f3c5-00bb-4d7e-8867-f3c038f43896
# ╟─549b29d8-ac43-4b83-beda-76b95e686425
# ╟─c539c5f0-2f0f-4af2-a7b8-312366e5428a
# ╟─f4aff33a-0a9f-47e0-a509-0d0487cb119b
# ╠═99cce976-0674-4aa1-981e-f81ad12d6a8c
# ╠═0baa1ee7-2b20-4d42-a3c6-7ff4a3a703fa
# ╟─f1c4b57d-f091-4056-bf32-59b95b65240f
# ╠═d6fcac43-9b06-4891-a64b-026440cc0b70
# ╟─916ccdaf-1640-40e6-943a-e43298a6f0a5
# ╠═0b128573-d982-4ad9-8e95-72d376a376bc
# ╠═a395acde-5336-499f-94e5-015efbe7564f
# ╠═9885c6e0-2c61-41f9-ad96-5660e4be02ea
# ╟─59baf20f-7e10-4168-aab6-372a924a0368
# ╠═69be43d6-7260-4bea-a021-09aedede16d7
# ╠═521105b5-de12-4fa5-837a-8ba85cd7e4cb
# ╟─7e5edff7-fe95-4b38-9134-6a08cf6e7926
# ╟─55aef241-fd15-4dfb-b8d7-7980d7cc0658
# ╟─10038161-9f4e-478a-916d-56e5f709f366
# ╠═651b85f4-2b07-421f-925d-2203ec0e5795
# ╟─55e04369-57b0-4583-95e6-7c3a030a9a58
# ╟─3fee9d47-39a5-42c5-87ac-e4ded1f15d40
# ╠═c8894712-1b6a-4b37-98de-16c071e9b207
# ╠═35c751ad-6943-4919-8301-6e3d5bb32b23
# ╟─ca827adf-cc47-45cf-9b4b-627b4668016f
# ╟─a4d71dc7-77f2-40a6-a55f-b92c666e2497
# ╠═8a113e7a-1501-4e81-8c03-e6e49803d8df
# ╟─6c48af93-a204-4888-93ea-434e80025aed
# ╠═a939ebe8-0608-42f9-96c5-c4764e1484c7
# ╟─3a8bc0ae-946d-4004-a038-afcd989a3e25
# ╟─7ec8f015-d923-4dfd-acc5-ea1768987cd5
# ╟─33daf7d0-054c-463b-9399-dfc1f5e8d5b4
# ╟─3e444d85-c9c1-49ea-9f28-a6c65219cc52
# ╟─da9e5ec9-db5a-441a-b6a4-b70f80e9bdcf
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
