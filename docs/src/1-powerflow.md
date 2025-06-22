```@meta
EditURL = "../../1-powerflow.jl"
```

# Tutorial 1: solving the power-flow equations with ExaModels

In this tutorial, we detail how to use ExaModels to solve the power flow
equations on the GPU. We start by describing the model we use, and then write
a basic Newton solver in Julia. Then we detail how to port the algorithm to the GPU
for faster performance

We start by importing the usual packages (including JLD2, a package to import
serialized data in Julia)

````@example 1-powerflow
using LinearAlgebra
using SparseArrays

using NLPModels
using ExaModels

using JLD2

include("utils.jl")
````

We load the classical case9ieee instance, here generated using the MATPOWER
file found in [matpower repo](https://github.com/MATPOWER/).

````@example 1-powerflow
DATA_DIR = "/home/fpacaud/dev/examodels-tutorials/instances"
data = JLD2.load(joinpath(DATA_DIR, "case9.jld2"))["data"]
````

The number of buses, generators and lines are:

````@example 1-powerflow
nbus = length(data.bus)
ngen = length(data.gen)
nlines = length(data.branch)
````

We load the indexes of the PV buses and the generators at the PV buses:

````@example 1-powerflow
pv_buses = get_pv_buses(data)
free_gen = get_free_generators(data)
````

## Implementing the power flow equations with ExaModels

When using the polar formulation, the power flow model requires the following variables:

1. The voltage magnitude at nodes ``v_m``
2. The voltage angles at nodes ``v_a``
3. The active power generation ``p_g``
4. The reactive power generation ``q_g``
5. The active power flow through the lines ``p``
6. The reactive power flow through the lines ``q``

The variables ``p`` and ``q`` are dependent variables depending on the voltage magnitudes
and angles at the adjacent nodes. The structure of the problem implies that the only
degree-of-freedom are the voltage magnitude at the PV and REF buses, the voltage angle at the REF buses
(usually set equal to 0) and the active power generation at the PV buses.

We define the variable in ExaModels.

````@example 1-powerflow
core = ExaCore()
va = variable(core, nbus)
vm = variable(core, nbus; start = data.vm0)
pg = variable(core, ngen; start=data.pg0)
qg = variable(core, ngen; start=data.qg0)
p = variable(core, 2*nlines) # FR and TO lines
q = variable(core, 2*nlines) # FR and TO lines
````

We set the initial values in `vm`, `pg` and `qg` using the setpoint values
specified in the matpower file.

As we solve the power flow equations, the degree-of-freedom are fixed. We fix them
in the model using a set of equality constraints:
We iterate over the reference buses to set their voltage and to 0

````@example 1-powerflow
c1 = constraint(core, va[i] for i in data.ref_buses)
````

and over the PV buses to set the voltage magnitude to the setpoint

````@example 1-powerflow
c01 = constraint(core, vm[i] for i in pv_buses; lcon=data.vm0[pv_buses], ucon=data.vm0[pv_buses])
````

and finally over the generators to fix the active power generation (except at the REF buses):

````@example 1-powerflow
c02 = constraint(core, pg[i] for i in free_gen; lcon=data.pg0[free_gen], ucon=data.pg0[free_gen])
````

We use the same model as in [MATPOWER](https://matpower.org/docs/manual.pdf)
to model the transmission lines, based on the standard ``π`` transmission line model in series with an ideal phase-shifting transformer.
Using the polar formulation, the active power through the line ``(i, j)`` is defined as
```math
  p_{i j} = g_{i i} v_{m,i}^2
  + g_{i j} v_{m, i} v_{m, j} \cos(v_{a, i} - v_{a, j})
  + b_{i j} v_{m, i} v_{m, j} \sin(v_{a, i} - v_{a, j})

```
and the reactive power is defined similarly as
```math
  q_{i j} = g_{i i} v_{m,i}^2
  + g_{i j} v_{m, i} v_{m, j} \sin(v_{a, i} - v_{a, j})
  - b_{i j} v_{m, i} v_{m, j} \cos(v_{a, i} - v_{a, j})

```
Using ExaModels, these two equations translate to the following constraints at the origin

````@example 1-powerflow
c2 = ExaModels.constraint(
    core,
    p[b.f_idx] - b.c5 * vm[b.f_bus]^2 -
    b.c3 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
    b.c4 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])) for
    b in data.branch
)
c3 = ExaModels.constraint(
    core,
    q[b.f_idx] +
    b.c6 * vm[b.f_bus]^2 +
    b.c4 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
    b.c3 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])) for
    b in data.branch
)
````

Similarly, the power flow at the destination are

````@example 1-powerflow
c4 = ExaModels.constraint(
    core,
    p[b.t_idx] - b.c7 * vm[b.t_bus]^2 -
    b.c1 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
    b.c2 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])) for
    b in data.branch
)
````

Reactive power flow, TO

````@example 1-powerflow
c5 = ExaModels.constraint(
    core,
    q[b.t_idx] +
    b.c8 * vm[b.t_bus]^2 +
    b.c2 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
    b.c1 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])) for
    b in data.branch
)
````

It remains to write the power flow balance equations at each bus.
They are defined, for the active power flow at bus ``i``
```math
   p_{g, i} - p_{d, i} - g_{s,i} v_{m,i}^2 = ∑_{j ∈ N(i)} p_{ij}

```
and for the reactive power flo at bus ``i``
```math
   q_{g, i} - q_{d, i} - b_{s,i} v_{m,i}^2 = ∑_{j ∈ N(i)} q_{ij}

```

Note that both set of constraints require to sum over the power flow at the adjacent lines.
As we have seen before, ExaModels defines the sum with a reduction over a given iterator.
As a consequence, we will evaluate the first terms ``p_{g, i} - p_{d, i} - g_{s,i} v_{m,i}^2 ``
apart from the sum ``∑_{j ∈ N(i)} p_{ij}`` in the expression tree defining the active
power flow balance. This translates to the following syntax in ExaModels. We first
iterate over all the buses to define the first part of the expressions:

````@example 1-powerflow
active_flow_balance = ExaModels.constraint(core, b.pd + b.gs * vm[b.i]^2 for b in data.bus)
````

Then we modify the constraint inplace to add the contribution of the adjacent lines

````@example 1-powerflow
ExaModels.constraint!(core, active_flow_balance, a.bus => p[a.i] for a in data.arc)
````

and finally, we add the contribution of the generators connected to each bus:

````@example 1-powerflow
ExaModels.constraint!(core, active_flow_balance, g.bus => -pg[g.i] for g in data.gen)
````

We follow the same procedure for the reactive power flow balance:

````@example 1-powerflow
reactive_flow_balance = ExaModels.constraint(core, b.qd - b.bs * vm[b.i]^2 for b in data.bus)
ExaModels.constraint!(core, reactive_flow_balance, a.bus => q[a.i] for a in data.arc)
ExaModels.constraint!(core, reactive_flow_balance, g.bus => -qg[g.i] for g in data.gen)
````

We have now defined all the equations needed to evaluate the power flow equations!
Note that we have defined all the expressions inside ExaModels: to evaluate them,
we convert the ExaCore to a proper ExaModel as:

````@example 1-powerflow
nlp = ExaModel(core)
````

Using NLPModels, evaluating the power flow at the initial setpoint amounts to

````@example 1-powerflow
x0 = NLPModels.get_x0(nlp)
c = NLPModels.cons(nlp, x0)
````

Remember that the first equations `c1`, `c01`, and `c02` are fixing the degree-of-freedom
to their setpoint. The power flow equations per-se are defined by the remaining equations,
starting with the constraint `c2`:

````@example 1-powerflow
m_fixed = c2.offset # use offset to determine where does the power flow eq. start in the model
````

We compute the norm-2 of the initial residual:

````@example 1-powerflow
residual = norm(c[m_fixed+1:end])
````

Note that if the power flow equations are satisfied, this residual should be 0, which is not
the case here. We remember that our degree-of-freedom are:

- voltage angle at ref buses;
- voltage magnitude at PV and ref buses;
- active power generation at PV buses;

We keep the degree-of-freedom fixed, and looks for the dependent variables
satisfying the power flow equations for this given setpoint. To do this, we will
use Newton method over the power flow balance equations.

## Solving the power flow equations using the Newton algorithm

We load the numbers of variables, constraints and nonzeros in the Jacobian
(all these values are provided automatically by ExaModels):

````@example 1-powerflow
n = NLPModels.get_nvar(nlp)
m = NLPModels.get_ncon(nlp)
nnzj = NLPModels.get_nnzj(nlp)
````

We load the index of the degree-of-freedom in our model using a utility function:

````@example 1-powerflow
ind_dof = get_index_dof(data)
````

and the indexes of dependent variables are automatically defined as

````@example 1-powerflow
ind_dep = setdiff(1:n, ind_dof)
````

We start by evaluating the Jacobian of our model using NLPModels syntax.
We get the sparsity pattern of our Jacobian in COO format directly by using:

````@example 1-powerflow
Ji = similar(x0, Int, nnzj)
Jj = similar(x0, Int, nnzj)
NLPModels.jac_structure!(nlp, Ji, Jj)
````

and we evaluate the nonzero values using

````@example 1-powerflow
Jx = similar(x0, nnzj)
NLPModels.jac_coord!(nlp, x0, Jx)
````

Julia uses the CSC format by default to store sparse matrix. We can convert
our Jacobian to CSC directly using Julia syntax:

````@example 1-powerflow
J = sparse(Ji, Jj, Jx, m, n)
````

And we can extract from the Jacobian the part associated to the power flow balance;

````@example 1-powerflow
G = J[m_fixed+1:end, ind_dep]
````

This is the matrix we need in the Newton algorithm. But before implementing it, we need
one last routine to pass the data from the vector `Jx` (in COO format) to the nonzeros
in the CSC matrix G. To do this, we use the following trick:

````@example 1-powerflow
Jx .= 1:nnzj # store index of each coefficient in Jx
J = sparse(Ji, Jj, Jx, m, n)  # convert the COO matrix to CSC
G = J[m_fixed+1:end, ind_dep] # extract the submatrix associated to the power flow equations
coo_to_csc = convert.(Int, nonzeros(G))
````

Using this vector of indices, we can automatically pass the data from Jx to G with:

````@example 1-powerflow
nonzeros(G) .= Jx[coo_to_csc]
````

We are now in place to solve the power flow equations. We start by importing KLU:

````@example 1-powerflow
using KLU
````

and we initialize the Newton algorithm by evaluating the model at the initial point:

````@example 1-powerflow
x = copy(x0)
c = similar(x0, m)
d = similar(x0, length(ind_dep))     # descent direction
residual = view(c, m_fixed+1:m)      # get subvector associated to the power flow residual

NLPModels.cons!(nlp, x, c)
NLPModels.jac_coord!(nlp, x, Jx)
nonzeros(G) .= Jx[coo_to_csc]
````

We compute the symbolic factorization using the direct solver KLU directly as

````@example 1-powerflow
ls = klu(G)
````

The Newton algorithm writes:

````@example 1-powerflow
max_iter = 10
tol = 1e-8

@info "Solving the power flow equations with Newton"
for i in 1:max_iter
    @info "It: $(i) residual: $(norm(residual))"
    if norm(residual) <= tol
        break
    end
    NLPModels.jac_coord!(nlp, x, Jx) # Update values in Jacobian
    nonzeros(G) .= Jx[coo_to_csc]
    klu!(ls, G)                      # Update numerical factorization
    ldiv!(d, ls, residual)           # Compute Newton direction using a backsolve
    x[ind_dep] .-= d
    NLPModels.cons!(nlp, x, c)
end
````

We observe that the Newton algorithm has converged in 5 iterations! The final
residual is not exactly 0 but is close enough (close to 1e-14).
We can recover the solution directly by looking at the values in the vector `x`:

````@example 1-powerflow
va_sol = x[1:nbus]
vm_sol = x[nbus+1:2*nbus]
````

We implement the generation of the model in a function `powerflow_model`,
and the Newton algorithm in another function `solve_power_flow`:

````@example 1-powerflow
include("powerflow.jl")
````

You can test the performance of Newton on various cases using the following code:

````@example 1-powerflow
data = JLD2.load(joinpath(DATA_DIR, "pglib_opf_case1354_pegase.jld2"))["data"]
ngen = length(data.gen)
nbus = length(data.bus)
nlines = length(data.branch)

nlp = powerflow_model(data)
results = solve_power_flow(nlp)
nothing
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

