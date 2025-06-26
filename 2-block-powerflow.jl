
# # Tutorial 2: solving the power-flow equations in batch on the GPU
#
# In the previous tutorial, we have seen how to solve the power flow equations
# using ExaModels. Now, we want to fully leverage the capability of ExaModels
# to solve the power flow equations in batch on the GPU.
#
# We start by importing the usual packages:
#
using LinearAlgebra
using SparseArrays

using NLPModels
using ExaModels

using JLD2

include("utils.jl")

# We load again the instance case9ieee:
DATA_DIR = joinpath(splitdir(Base.active_project())[1], "instances")
data = JLD2.load(joinpath(DATA_DIR, "case9.jld2"))["data"]

nbus = length(data.bus)
ngen = length(data.gen)
nlines = length(data.branch)

# ## Block power flow with ExaModels
#
# The power flow are parameterized by the active and reactive power loads
# ``p_d`` and ``q_d`` at each bus, among others. This gives a total
# of `2*nbus` parameters.
#
# In this tutorial, we want to solve the power flow equations in batch for ``N``
# different load scenarios ``\{ p_d^n, q_d^n \}_{n=1,⋯,N}``, with ``N`` playing the role of the batch size.
# To each scenario ``(p_d^n, q_d^n)`` is associated a block. The number of blocks is the batch size ``N``.

# As a demonstration, we set the batch size to 100:
N = 100

# For each parameter ``(p_d^n, q_d^n)`` is associated a given solution
# ``(v_m^n, v_a^n, p_g^n, q_g^n)`` of the power flow equations. We will look at computing
# all the solutions in parallel using Newton.

# Using ExaModels, we can define the corresponding block model by augmenting the
# dimension of each variable with a second dimension parameterized by the batch size ``N``.
# This amounts to define the following variables:
core = ExaCore()
va = variable(core, nbus, 1:N)
vm = variable(core, nbus, 1:N; start = repeat(data.vm0, N))
pg = variable(core, ngen, 1:N;  start=repeat(data.pg0, N))
qg = variable(core, ngen, 1:N;  start=repeat(data.qg0, N))
p = variable(core, 2*nlines, 1:N)
q = variable(core, 2*nlines, 1:N)

# Note that we have to duplicate ``N`` times the starting point for ``v_m``, ``p_g`` and ``q_g``.
# We also have to evaluate the power flow constraint in block. As a consequence, the iterator
# used to generate each constraint has to be modified using the iterator `product`:
c2 = constraint(
    core,
    p[b.f_idx, k]
    - b.c5 * vm[b.f_bus, k]^2 -
    b.c3 * (vm[b.f_bus, k] * vm[b.t_bus, k] * cos(va[b.f_bus, k] - va[b.t_bus, k])) -
    b.c4 * (vm[b.f_bus, k] * vm[b.t_bus, k] * sin(va[b.f_bus, k] - va[b.t_bus, k])) for
    (b, k) in product(data.branch, 1:N)
)

# To avoid redefining all the models, we provide a utility function to generate the
# block power flow model using ExaModels:
include("powerflow.jl")

nlp = block_power_flow_model(data, N)

# The power flow model can be solved on the CPU using the function
# `solve_power_flow` we implemented in the previous tutorial:
results = solve_power_flow(nlp, N)
nothing

# We recover the solution in matrix format using:
vm = reshape(results[nbus*N+1:2*nbus*N], nbus, N)

# ## Solving the power flow equations in batch on the GPU
#
# Note that we don't exploit in the algorithm the fact that the ``N`` blocks are independent.
# ExaModels is able to detect the repeated data structure automatically, and can evaluate the resulting model in
# parallel on the GPU. That's the core benefit of the SIMD abstraction used by ExaModels.
# To evaluate the model on the GPU using ExaModels, you just have to pass the correct backend
# to the function `block_power_flow_model` we used just before:
using CUDA
nlp_gpu = block_power_flow_model(data, N; backend=CUDABackend())

n = NLPModels.get_nvar(nlp_gpu)
m = NLPModels.get_ncon(nlp_gpu)
nnzj = NLPModels.get_nnzj(nlp_gpu)

# Evaluating the model on the GPU simply amounts to
x0 = NLPModels.get_x0(nlp_gpu)
c = similar(x0, m)
NLPModels.cons!(nlp_gpu, x0, c)
# for the power flow residual, and for the Jacobian:
Jx = similar(x0, nnzj)
NLPModels.jac_coord!(nlp_gpu, x0, Jx)

# We can benchmark the time spent in the evaluation of the derivative
# using the macro `@time` (or `CUDA.@time` if we want also to include the
# synchronization time in CUDA):

CUDA.@time NLPModels.cons!(nlp_gpu, x0, c)
CUDA.@time NLPModels.jac_coord!(nlp_gpu, x0, Jx)
nothing

# We observe that the evaluation of the Jacobian takes 0.3ms in this case.
# In the function `analyse_sparsity`, we provide a sparse routine extracting the submatrix corresponding to the power flow equations
# from the Jacobian J.
#
# !!! info
#     On the GPU, the default format for sparse matrices is CSR, as it
#     leads to better parallelism when computing sparse-matrix vector products.

# We can assemble the submatrix `G` using this new function:
ind_dof = get_index_dof(data, N)
m_fixed = length(ind_dof)
ind_dep = setdiff(1:n, ind_dof)
nx = length(ind_dep)

Ji = similar(x0, Int, nnzj)
Jj = similar(x0, Int, nnzj)
NLPModels.jac_structure!(nlp_gpu, Ji, Jj)

G, coo_to_csr = analyse_sparsity(Ji, Jj, Jx, m, n, m_fixed, ind_dep)

# Now the Jacobian is evaluated, we have to compute the LU factorization on the GPU,
# if possible in sparse format. The solver [cuDSS](https://docs.nvidia.com/cuda/cudss/getting_started.html) allows to do exactly that. To use cuDSS in Julia, you have to import the package CUDSS

using CUDSS

# We update the values in the Jacobian of the original model and transfer them to `G`
# using the same procedure as in Tutorial 1:

NLPModels.jac_coord!(nlp_gpu, x0, Jx)
nonzeros(G) .= Jx[coo_to_csr]

# The symbolic factorization in cuDSS proceeds as follows:

d_gpu = CUDA.zeros(Float64, nx)
b_gpu = CUDA.zeros(Float64, nx)

solver = CudssSolver(G, "G", 'F')
cudss_set(solver, "reordering_alg", "algo2") # we have to change the ordering to get valid results
cudss("analysis", solver, d_gpu, b_gpu)

# Hence, we are now able to replace KLU by CUDSS in the Newton solver we implemented
# in the previous tutorial.
# We initialize the Newton algorithm by evaluating the model at the initial point:
ind_dep = CuVector{Int}(ind_dep)
x = copy(x0)
c = similar(x0, m)
residual = view(c, m_fixed+1:m)      # get subvector associated to the power flow residual

NLPModels.cons!(nlp_gpu, x, c)

cudss("factorization", solver, d_gpu, b_gpu)

max_iter = 10
tol = 1e-8

@info "Solving the power flow equations with Newton"
i = 1
for i in 1:max_iter
    @info "It: $(i) residual: $(norm(residual))"
    if norm(residual) <= tol
        break
    end
    NLPModels.jac_coord!(nlp_gpu, x, Jx) # Update values in Jacobian
    nonzeros(G) .= Jx[coo_to_csr]
    cudss_set(solver, G)                 # Update numerical factorization
    cudss("refactorization", solver, d_gpu, b_gpu)
    b_gpu .= residual
    cudss("solve", solver, d_gpu, b_gpu)
    x[ind_dep] .-= d_gpu
    NLPModels.cons!(nlp_gpu, x, c)
end

# We observe that we get exactly the same convergence as before on the CPU.
# However, the time to solution is significantly higher than on the CPU: it turns out that
# KLU is much more efficient than cuDSS on this particular example.


