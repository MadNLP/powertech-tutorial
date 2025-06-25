
using LinearAlgebra
using SparseArrays
using KLU

using CUDA
using CUDA.CUSPARSE
using CUDSS

function powerflow_model(
    data;
    backend = nothing,
    T = Float64,
    kwargs...,
)

    ngen = length(data.gen)
    nbus = length(data.bus)
    nlines = length(data.branch)

    pv_buses = get_pv_buses(data)
    free_gen = get_free_generators(data)

    w = ExaModels.ExaCore(T; backend = backend)
    va = ExaModels.variable(w, nbus)
    vm = ExaModels.variable(
        w,
        nbus;
        start = data.vm0,
    )
    pg = ExaModels.variable(w, ngen;  start=data.pg0)
    qg = ExaModels.variable(w, ngen;  start=data.qg0)
    p = ExaModels.variable(w, 2*nlines)
    q = ExaModels.variable(w, 2*nlines)

    # Fix variables to setpoint
    c1 = ExaModels.constraint(w, va[i] for i in data.ref_buses)
    c01 = ExaModels.constraint(w, vm[i] for i in pv_buses; lcon=data.vm0[pv_buses], ucon=data.vm0[pv_buses])
    c02 = ExaModels.constraint(w, pg[i] for i in free_gen; lcon=data.pg0[free_gen], ucon=data.pg0[free_gen])

    # Active power flow, FR
    c2 = ExaModels.constraint(
        w,
        p[b.f_idx] - b.c5 * vm[b.f_bus]^2 -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
        b.c4 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])) for
        b in data.branch
    )
    # Reactive power flow, FR
    c3 = ExaModels.constraint(
        w,
        q[b.f_idx] +
        b.c6 * vm[b.f_bus]^2 +
        b.c4 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])) for
        b in data.branch
    )
    # Active power flow, TO
    c4 = ExaModels.constraint(
        w,
        p[b.t_idx] - b.c7 * vm[b.t_bus]^2 -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
        b.c2 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])) for
        b in data.branch
    )
    # Reactive power flow, TO
    c5 = ExaModels.constraint(
        w,
        q[b.t_idx] +
        b.c8 * vm[b.t_bus]^2 +
        b.c2 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])) for
        b in data.branch
    )

    # Power flow constraints
    c9 = ExaModels.constraint(w, b.pd + b.gs * vm[b.i]^2 for b in data.bus)
    c10 = ExaModels.constraint(w, b.qd - b.bs * vm[b.i]^2 for b in data.bus)
    c11 = ExaModels.constraint!(w, c9, a.bus => p[a.i] for a in data.arc)
    c12 = ExaModels.constraint!(w, c10, a.bus => q[a.i] for a in data.arc)
    c13 = ExaModels.constraint!(w, c9, g.bus => -pg[g.i] for g in data.gen)
    c14 = ExaModels.constraint!(w, c10, g.bus => -qg[g.i] for g in data.gen)

    return ExaModels.ExaModel(w; kwargs...)
end

function analyse_sparsity(Ji::Vi, Jj::Vi, Jx::Vx, m::Int, n::Int, m_fixed, ind_dep) where {Vi <: Array{<:Integer}, Vx <: Array{<: AbstractFloat}}
    nnzj = length(Ji)
    Jx .= 1:nnzj
    J = sparse(Ji, Jj, Jx, m, n)
    G = J[m_fixed+1:end, ind_dep]
    coo_to_csc = convert.(Int, nonzeros(G))
    return G, coo_to_csc
end

function solve_power_flow(nlp, N=1; tol=1e-8, max_iter=10)
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    nnzj = NLPModels.get_nnzj(nlp)
    x0 = NLPModels.get_x0(nlp)

    ind_dof = get_index_dof(data, N)
    m_fixed = length(ind_dof)
    ind_dep = setdiff(1:n, ind_dof)

    Ji = similar(x0, Int, nnzj)
    Jj = similar(x0, Int, nnzj)
    NLPModels.jac_structure!(nlp, Ji, Jj)

    Jx = similar(x0, nnzj)

    G, coo_to_csc = analyse_sparsity(Ji, Jj, Jx, m, n, m_fixed, ind_dep)

    # Initialize data structure
    x = copy(x0)
    c = similar(x0, m)
    d = similar(x0, length(ind_dep))     # descent direction
    residual = view(c, m_fixed+1:m)      # get subvector associated to the power flow residual

    NLPModels.cons!(nlp, x, c)
    NLPModels.jac_coord!(nlp, x, Jx)
    nonzeros(G) .= Jx[coo_to_csc]

    ls = klu(G)

    for i in 1:max_iter
        @info "It: $(i) residual: $(norm(residual))"
        # Stopping criterion
        if norm(residual) <= tol
            break
        end
        # Update values in Jacobian
        NLPModels.jac_coord!(nlp, x, Jx)
        nonzeros(G) .= Jx[coo_to_csc]
        # Update numerical factorization
        klu!(ls, G)
        # Compute Newton direction using a backsolve
        ldiv!(d, ls, residual)
        # Update dependent variables
        x[ind_dep] .-= d
        # Refresh residuals
        NLPModels.cons!(nlp, x, c)
    end

    return x
end

function block_power_flow_model(
    data, K;
    backend = nothing,
    T = Float64,
    kwargs...,
)

    ngen = length(data.gen)
    nbus = length(data.bus)
    nlines = length(data.branch)

    idx_bus = [(b, k) for b in 1:nbus, k in 1:K]

    pv_buses = get_pv_buses(data)
    free_gen = get_free_generators(data)

    w = ExaModels.ExaCore(T; backend = backend)
    va = ExaModels.variable(w, nbus, 1:K)
    vm = ExaModels.variable(
        w,
        nbus, 1:K;
        start = repeat(data.vm0, K)
    )
    pg = ExaModels.variable(w, ngen, 1:K;  start=repeat(data.pg0, K))
    qg = ExaModels.variable(w, ngen, 1:K;  start=repeat(data.qg0, K))
    p = ExaModels.variable(w, 2*nlines, 1:K)
    q = ExaModels.variable(w, 2*nlines, 1:K)

    # Fix variables to setpoint
    c1 = ExaModels.constraint(
        w,
        va[i, k] for
        (i, k) in product(data.ref_buses, 1:K)
    )
    c01 = ExaModels.constraint(
        w,
        vm[i, k] for
        (i, k) in product(pv_buses, 1:K);
        lcon=repeat(data.vm0[pv_buses], K),
        ucon=repeat(data.vm0[pv_buses], K),
    )
    c02 = ExaModels.constraint(
        w,
        pg[i, k] for
        (i, k) in product(free_gen, 1:K);
        lcon=repeat(data.pg0[free_gen], K),
        ucon=repeat(data.pg0[free_gen], K),
    )

    # Active power flow, FR
    c2 = ExaModels.constraint(
        w,
        p[b.f_idx, k]
        - b.c5 * vm[b.f_bus, k]^2 -
        b.c3 * (vm[b.f_bus, k] * vm[b.t_bus, k] * cos(va[b.f_bus, k] - va[b.t_bus, k])) -
        b.c4 * (vm[b.f_bus, k] * vm[b.t_bus, k] * sin(va[b.f_bus, k] - va[b.t_bus, k])) for
        (b, k) in product(data.branch, 1:K)
    )
    # Reactive power flow, FR
    c3 = ExaModels.constraint(
        w,
        q[b.f_idx, k] +
        b.c6 * vm[b.f_bus, k]^2 +
        b.c4 * (vm[b.f_bus, k] * vm[b.t_bus, k] * cos(va[b.f_bus, k] - va[b.t_bus, k])) -
        b.c3 * (vm[b.f_bus, k] * vm[b.t_bus, k] * sin(va[b.f_bus, k] - va[b.t_bus, k])) for
        (b, k) in product(data.branch, 1:K)
    )
    # Active power flow, TO
    c4 = ExaModels.constraint(
        w,
        p[b.t_idx, k]
        - b.c7 * vm[b.t_bus, k]^2 -
        b.c1 * (vm[b.t_bus, k] * vm[b.f_bus, k] * cos(va[b.t_bus, k] - va[b.f_bus, k])) -
        b.c2 * (vm[b.t_bus, k] * vm[b.f_bus, k] * sin(va[b.t_bus, k] - va[b.f_bus, k])) for
        (b, k) in product(data.branch, 1:K)
    )
    # Reactive power flow, TO
    c5 = ExaModels.constraint(
        w,
        q[b.t_idx, k] +
        b.c8 * vm[b.t_bus, k]^2 +
        b.c2 * (vm[b.t_bus, k] * vm[b.f_bus, k] * cos(va[b.t_bus, k] - va[b.f_bus, k])) -
        b.c1 * (vm[b.t_bus, k] * vm[b.f_bus, k] * sin(va[b.t_bus, k] - va[b.f_bus, k])) for
        (b, k) in product(data.branch, 1:K)
    )

    perturb = randn(nbus, K)
    # Power flow constraints
    c9 = ExaModels.constraint(w, b.pd + b.gs * vm[b.i, k]^2 for (b, k) in product(data.bus, 1:K))
    c10 = ExaModels.constraint(w, b.qd - b.bs * vm[b.i, k]^2 for (b, k) in product(data.bus, 1:K))
    c11 = ExaModels.constraint!(w, c9, a.bus + (k-1)*nbus => p[a.i, k] for (a, k) in product(data.arc, 1:K))
    c12 = ExaModels.constraint!(w, c10, a.bus + (k-1)*nbus => q[a.i, k] for (a, k) in product(data.arc, 1:K))
    c13 = ExaModels.constraint!(w, c9, g.bus + (k-1)*nbus => -pg[g.i, k] for (g, k) in product(data.gen, 1:K))
    c14 = ExaModels.constraint!(w, c10, g.bus + (k-1)*nbus => -qg[g.i, k] for (g, k) in product(data.gen, 1:K))

    return ExaModels.ExaModel(w; kwargs...)
end


function analyse_sparsity(Ji::Vi, Jj::Vi, Jx::Vx, m::Int, n::Int, m_fixed, ind_dep) where {Vi <: CuArray{<:Integer}, Vx <: CuArray{<: AbstractFloat}}
    nnzj = length(Ji)

    # Perform the analysis on the GPU
    Ji_ = Array(Ji)
    Jj_ = Array(Jj)
    Jx_ = Array(Jx)
    Jx_ .= 1:nnzj

    J_ = sparse(Ji_, Jj_, Jx_, m, n)
    G_ = J_[m_fixed+1:end, ind_dep]
    G = CuSparseMatrixCSR(G_)
    coo_to_csr = convert.(Int, nonzeros(G))
    return G, CuArray{Int}(coo_to_csr)
end


nothing
