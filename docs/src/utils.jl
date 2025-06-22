
import Base.Iterators: product

convert_data(data::N, backend) where {names,N<:NamedTuple{names}} =
    NamedTuple{names}(ExaModels.convert_array(d, backend) for d in data)

# Get all PV buses
function get_pv_buses(data)
    return [b.i for b in data.bus if b.bus_type ∈ [2, 3]]
end
# Get all PQ buses
function get_pq_buses(data)
    return [b.i for b in data.bus if b.bus_type == 1]
end
# Get all reference buses
function get_ref_buses(data)
    return [b.i for b in data.bus if b.bus_type == 3]
end
# Get generators associated to PV nodes
function get_free_generators(data)
    return [g.i for g in data.gen if data.bus[g.bus].bus_type == 2]
end
# Return fixed variables in power flow problem.
function get_index_dof(data, K=1)
    nbus = length(data.bus)
    ngen = length(data.gen)
    # Degrees-of-freedom are:
    # 1. voltage angle at reference buses
    ref = get_ref_buses(data)
    # 2. voltage magnitude at reference and PV buses
    pv = get_pv_buses(data)
    # 3. active power generation at PV buses
    gen = get_free_generators(data)

    n_dof = length(ref) + length(pv) + length(gen)

    fixed_variables = zeros(Int, n_dof * K)

    shift_vm = nbus * K
    shift_gen = 2 * nbus * K

    cnt = 0
    for k in 1:K, i in ref
        fixed_variables[cnt += 1] = i + (k-1)*nbus
    end
    for k in 1:K, i in pv
        fixed_variables[cnt += 1] = i + (k-1)*nbus + shift_vm
    end
    for k in 1:K, i in gen
        fixed_variables[cnt += 1] = i + (k-1)*ngen + shift_gen
    end

    return fixed_variables
end
function get_block_reordering(data, K)
    nbus = length(data.bus)
    ngen = length(data.gen)
    nlines = length(data.branch)


    nvar = (2*nbus + 2*ngen + 4*nlines) * K
    shift_vm = nbus * K
    shift_pg = 2 * nbus * K
    shift_qg = 2 * nbus * K + ngen * K
    shift_p = 2 * nbus * K + 2 * ngen * K

    perm = zeros(Int, nvar)

    cnt = 0
    for k in 1:K
        # Va
        for i in 1:nbus
            perm[cnt += 1] = i + (k-1)*nbus
        end
        # Vm
        for i in 1:nbus
            perm[cnt += 1] = i + (k-1)*nbus + shift_vm
        end
        # Pg
        for i in 1:ngen
            perm[cnt += 1] = i + (k-1)*ngen + shift_pg
        end
        # Qg
        for i in 1:ngen
            perm[cnt += 1] = i + (k-1)*ngen + shift_qg
        end
        # P
        for i in 1:2*nlines
            perm[cnt += 1] = i + (k-1)*2*nlines + shift_p
        end
        # Q
        for i in 1:2*nlines
            perm[cnt += 1] = i + (k-1)*2*nlines + shift_p + 2*nlines*K
        end
    end

    return perm
end


