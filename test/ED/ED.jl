using LinearAlgebra, SparseArrays

# State ∈ [0, up, down, updown] = [00, 10, 01, 11]
# 2x2 lattice -> 4 (exponent) -> 4^4 states = 256
const UP = 1
const DOWN = 2

function state_from_integer(int, sites=4, substates_per_site=2)
    int > (2substates_per_site)^sites-1 && return
    out = BitArray(undef, (substates_per_site, sites))
    out.chunks[1] = int
    out
end

function create(state, site, substate)
    # No state case
    state == 0 && return 0, 0.0
    # create(|1⟩) -> no state
    state[substate, site] && return 0, 0.0

    # fermionic sign
    lindex = 2(site-1) + substate
    _sign = iseven(sum(state[1:lindex-1])) ? +1.0 : -1.0

    # create(|0⟩) -> |1⟩
    s = copy(state)
    s[substate, site] = true
    return _sign, s
end

function annihilate(state, site, substate)
    # No state
    state == 0 && return 0, 0.0
    # annihilate(|0⟩) -> no state
    !state[substate, site] && return 0, 0.0

    # fermionic sign
    lindex = 2(site-1) + substate
    _sign = iseven(sum(state[1:lindex-1])) ? +1.0 : -1.0

    # annihilate(|1⟩) -> |0⟩
    s = copy(state)
    s[substate, site] = false
    return _sign, s
end



function HamiltonMatrix(model::HubbardModelAttractive)
    lattice = model.l
    t = model.t
    U = model.U
    mu = model.mu

    H = zeros(Float64, 4^lattice.sites, 4^lattice.sites)

    # -t ∑_ijσ c_iσ^† c_jσ
    # +U ∑_i (n_i↑ - 1/2)(n_i↓ - 1/2)
    # -μ ∑_i n_i
    for i in 1:4^lattice.sites
        lstate = state_from_integer(i-1, lattice.sites)
        for j in 1:4^lattice.sites
            rstate = state_from_integer(j-1, lattice.sites)

            E = 0
            # hopping (hermitian conjugate implied/included by lattice generation)
            for substate in [1, 2]
                for source in 1:lattice.sites
                    for target in lattice.neighs[:, source]
                        _sign1, state = annihilate(rstate, source, substate)
                        _sign2, state = create(state, target, substate)
                        if state != 0 && lstate == state
                            E -= _sign1 * _sign2 * t
                        end
                    end
                end
            end

            # # U, μ terms
            for p in 1:4
                up_occ = rstate[1, p]
                down_occ = rstate[2, p]
                if lstate == rstate
                    E += U * (up_occ - 0.5) * (down_occ - 0.5)
                    E -= mu * (up_occ + down_occ)
                end
            end

            H[i, j] = E
        end
    end
    H
end


function HamiltonMatrix(model::MonteCarlo.ZCModel)
    lattice = model.l
    t1x = model.t1x
    t1y = model.t1y
    t1z = model.t1z
    t2 = model.t2
    tperp = model.tperp
    U = model.U
    mu = model.mu

    UP = 1
    DOWN = 2

    H = zeros(ComplexF64, 4^lattice.sites, 4^lattice.sites)

    for i in 1:4^lattice.sites
        lstate = state_from_integer(i-1, lattice.sites)
        for j in 1:4^lattice.sites
            rstate = state_from_integer(j-1, lattice.sites)

            E = 0
            # hopping (hermitian conjugate implied/included by lattice generation)
            for source in 1:lattice.sites
                # X (A <-> B equal sign)
                for target in lattice.neighs[1:3:end, source]
                    lattice.isAsite[source] == lattice.isAsite[target] && error(
                        "$source and $target are on the same sublattice, σx, ED"
                    )
                    _sign1, state = annihilate(rstate, source, UP)
                    _sign2, state = create(state, target, UP)
                    if state != 0 && lstate == state
                        E += _sign1 * _sign2 * t1x
                    end
                    _sign1, state = annihilate(rstate, source, DOWN)
                    _sign2, state = create(state, target, DOWN)
                    if state != 0 && lstate == state
                        E += -1.0 * _sign1 * _sign2 * t1x
                    end
                end


                # Y (A <-> B differ)
                for target in lattice.neighs[2:3:end, source]
                    lattice.isAsite[source] == lattice.isAsite[target] && error(
                        "$source and $target are on the same sublattice, σy, ED"
                    )
                    AB_sign = lattice.isAsite[source] ? +1.0 : -1.0

                    _sign1, state = annihilate(rstate, source, UP)
                    _sign2, state = create(state, target, UP)
                    if state != 0 && lstate == state
                        E += AB_sign * 1im * _sign1 * _sign2 * t1y
                    end

                    _sign1, state = annihilate(rstate, source, DOWN)
                    _sign2, state = create(state, target, DOWN)
                    if state != 0 && lstate == state
                        E += -1.0 * AB_sign * 1im * _sign1 * _sign2 * t1y
                    end
                end


                # Z
                for target in lattice.neighs[3:3:end, source]
                    lattice.isAsite[source] != lattice.isAsite[target] && error(
                        "$source and $target are on different sublattices, σz, ED"
                    )
                    AB_sign = lattice.isAsite[source] ? +1.0 : -1.0

                    _sign1, state = annihilate(rstate, source, UP)
                    _sign2, state = create(state, target, UP)
                    if state != 0 && lstate == state
                        E += AB_sign * _sign1 * _sign2 * t1z
                    end

                    _sign1, state = annihilate(rstate, source, DOWN)
                    _sign2, state = create(state, target, DOWN)
                    if state != 0 && lstate == state
                        E += -1.0 * AB_sign * _sign1 * _sign2 * t1z
                    end
                end


                for target in lattice.ext_neighs[:, source]
                    lattice.isAsite[source] != lattice.isAsite[target] && error(
                        "$source and $target are on different sublattices, t2, ED"
                    )
                    _sign1, state = annihilate(rstate, source, UP)
                    _sign2, state = create(state, target, UP)
                    if state != 0 && lstate == state
                        E += _sign1 * _sign2 * t2
                    end

                    _sign1, state = annihilate(rstate, source, DOWN)
                    _sign2, state = create(state, target, DOWN)
                    if state != 0 && lstate == state
                        E += -1.0 * _sign1 * _sign2 * t2
                    end

                end

                _sign1, state = annihilate(rstate, source, DOWN)
                _sign2, state = create(state, source, UP)
                if state != 0 && lstate == state
                    E += _sign1 * _sign2 * tperp
                end
                # h.c
                _sign1, state = create(rstate, source, DOWN)
                _sign2, state = annihilate(state, source, UP)
                if state != 0 && lstate == state
                    E += _sign1 * _sign2 * tperp
                end
            end

            # # U, μ terms
            for p in 1:4
                up_occ = rstate[1, p]
                down_occ = rstate[2, p]
                if lstate == rstate
                    E += U * (up_occ - 0.5) * (down_occ - 0.5)
                    E += mu * (up_occ - down_occ)
                end
            end

            H[i, j] = E
        end
    end
    H
end


function HamiltonMatrix(model::MonteCarlo.ZCTModel)
    lattice = model.l
    t1x = model.t1x
    t1y = model.t1y
    t1z = model.t1z
    t2 = model.t2
    # tperp = model.tperp
    U = model.U

    UP = 1
    DOWN = 2

    H = zeros(ComplexF64, 4^lattice.sites, 4^lattice.sites)

    for i in 1:4^lattice.sites
        lstate = state_from_integer(i-1, lattice.sites)
        for j in 1:4^lattice.sites
            rstate = state_from_integer(j-1, lattice.sites)

            E = 0
            # hopping (hermitian conjugate implied/included by lattice generation)
            for source in 1:lattice.sites
                # X (A <-> B equal sign)
                for target in lattice.neighs[1:3:end, source]
                    lattice.isAsite[source] == lattice.isAsite[target] && error(
                        "$source and $target are on the same sublattice, σx, ED"
                    )
                    _sign1, state = annihilate(rstate, source, UP)
                    _sign2, state = create(state, target, UP)
                    if state != 0 && lstate == state
                        E += _sign1 * _sign2 * t1x
                    end
                    _sign1, state = annihilate(rstate, source, DOWN)
                    _sign2, state = create(state, target, DOWN)
                    if state != 0 && lstate == state
                        E += _sign1 * _sign2 * t1x
                    end
                end


                # Y (A <-> B differ)
                for target in lattice.neighs[2:3:end, source]
                    lattice.isAsite[source] == lattice.isAsite[target] && error(
                        "$source and $target are on the same sublattice, σy, ED"
                    )
                    AB_sign = lattice.isAsite[source] ? +1.0 : -1.0

                    _sign1, state = annihilate(rstate, source, UP)
                    _sign2, state = create(state, target, UP)
                    if state != 0 && lstate == state
                        E += AB_sign * 1im * _sign1 * _sign2 * t1y
                    end

                    _sign1, state = annihilate(rstate, source, DOWN)
                    _sign2, state = create(state, target, DOWN)
                    if state != 0 && lstate == state
                        E += AB_sign * 1im * _sign1 * _sign2 * t1y
                    end
                end


                # Z
                for target in lattice.neighs[3:3:end, source]
                    lattice.isAsite[source] != lattice.isAsite[target] && error(
                        "$source and $target are on different sublattices, σz, ED"
                    )
                    AB_sign = lattice.isAsite[source] ? +1.0 : -1.0

                    _sign1, state = annihilate(rstate, source, UP)
                    _sign2, state = create(state, target, UP)
                    if state != 0 && lstate == state
                        E += AB_sign * _sign1 * _sign2 * t1z
                    end

                    _sign1, state = annihilate(rstate, source, DOWN)
                    _sign2, state = create(state, target, DOWN)
                    if state != 0 && lstate == state
                        E += AB_sign * _sign1 * _sign2 * t1z
                    end
                end


                for target in lattice.ext_neighs[:, source]
                    lattice.isAsite[source] != lattice.isAsite[target] && error(
                        "$source and $target are on different sublattices, t2, ED"
                    )
                    _sign1, state = annihilate(rstate, source, UP)
                    _sign2, state = create(state, target, UP)
                    if state != 0 && lstate == state
                        E += _sign1 * _sign2 * t2
                    end

                    _sign1, state = annihilate(rstate, source, DOWN)
                    _sign2, state = create(state, target, DOWN)
                    if state != 0 && lstate == state
                        E += _sign1 * _sign2 * t2
                    end

                end

                # _sign1, state = annihilate(rstate, source, DOWN)
                # _sign2, state = create(state, source, UP)
                # if state != 0 && lstate == state
                #     E += _sign1 * _sign2 * tperp
                # end
                # # h.c
                # _sign1, state = create(rstate, source, DOWN)
                # _sign2, state = annihilate(state, source, UP)
                # if state != 0 && lstate == state
                #     E += _sign1 * _sign2 * tperp
                # end
            end

            # # U, μ terms
            for p in 1:4
                up_occ = rstate[1, p]
                down_occ = rstate[2, p]
                if lstate == rstate
                    E -= U * (up_occ - 0.5) * (down_occ - 0.5)
                end
            end

            H[i, j] = E
        end
    end
    H
end


# Greens function


function Greens(site1, site2, substate1, substate2)
    s -> begin
        _sign1, state = create(s, site1, substate1)
        state == 0 && return typeof(s)[], Float64[]
        _sign2, state = annihilate(state, site2, substate2)
        state == 0 && return typeof(s)[], Float64[]
        return [state], [_sign1 * _sign2]
    end
end

# According to p188 QMCM this is equivalent
# can be used to check if Greens/ED is correct
function Greens_permuted(site1, site2, substate1, substate2)
    s -> begin
        _sign1, state = annihilate(s, site2, substate2)
        _sign2, state = create(state, site1, substate1)
        delta = ((site1 == site2) && (substate1 == substate2)) ? 1.0 : 0.0

        if state == 0 && delta == 0.0
            # off-diagonal
            return typeof(s)[], Float64[]
        elseif state == 0
            # only delta function triggers (|0⟩ state)
            return [s], [delta]
        else
            # both trigger (|1⟩ state)
            return [state], [delta - _sign1 * _sign2]
        end
    end
end


# Charge Density Correlation
function charge_density_correlation(site1, site2)
    state -> begin
        states = typeof(state)[]
        prefactors = Float64[]
        for substate1 in [UP, DOWN]
            for substate2 in [UP, DOWN]
                sign1, _state = annihilate(state, site2, substate2)
                sign2, _state = create(_state, site2, substate2)
                sign3, _state = annihilate(_state, site1, substate1)
                sign4, _state = create(_state, site1, substate1)
                if _state != 0
                    push!(prefactors, sign1 * sign2 * sign3 * sign4)
                    push!(states, _state)
                end
            end
        end
        states, prefactors
    end
end


# Magnetization
function m_x(site)
    state -> begin
        states = typeof(state)[]
        prefactors = Float64[]
        _sign1, _state = annihilate(state, site, DOWN)
        _sign2, _state = create(_state, site, UP)
        if _state != 0
            push!(states, _state)
            push!(prefactors, _sign1 * _sign2)
        end
        _sign1, _state = annihilate(state, site, UP)
        _sign2, _state = create(_state, site, DOWN)
        if _state != 0
            push!(states, _state)
            push!(prefactors, _sign1 * _sign2)
        end
        return states, prefactors
    end
end
function m_y(site)
    state -> begin
        states = typeof(state)[]
        prefactors = Float64[]
        _sign1, _state = annihilate(state, site, DOWN)
        _sign2, _state = create(_state, site, UP)
        if _state != 0
            push!(states, _state)
            push!(prefactors, _sign1 * _sign2)
        end
        _sign1, _state = annihilate(state, site, UP)
        _sign2, _state = create(_state, site, DOWN)
        if _state != 0
            push!(states, _state)
            push!(prefactors, -1.0 * _sign1 * _sign2)
        end
        return states, -1im * prefactors
    end
end
function m_z(site)
    state -> begin
        states = typeof(state)[]
        prefactors = Float64[]
        _sign1, _state = annihilate(state, site, UP)
        _sign2, _state = create(_state, site, UP)
        if _state != 0
            push!(states, _state)
            push!(prefactors, _sign1 * _sign2)
        end
        _sign1, _state = annihilate(state, site, DOWN)
        _sign2, _state = create(_state, site, DOWN)
        if _state != 0
            push!(states, _state)
            push!(prefactors, -1.0 * _sign1 * _sign2)
        end
        return states, prefactors
    end
end



# Spin Density Correlations (s_{x, i} * s_{x, j} etc)
function spin_density_correlation_x(site1, site2)
    state -> begin
        states = typeof(state)[]
        prefactors = Float64[]
        for substates1 in [(UP, DOWN), (DOWN, UP)]
            for substates2 in [(UP, DOWN), (DOWN, UP)]
                sign1, _state = annihilate(state, site2, substates2[1])
                sign2, _state = create(_state, site2, substates2[2])
                sign3, _state = annihilate(_state, site1, substates1[1])
                sign4, _state = create(_state, site1, substates1[2])
                if _state != 0
                    push!(prefactors, sign1 * sign2 * sign3 * sign4)
                    push!(states, _state)
                end
            end
        end
        states, prefactors
    end
end
function spin_density_correlation_y(site1, site2)
    state -> begin
        states = typeof(state)[]
        prefactors = ComplexF64[]
        for substates1 in [(UP, DOWN), (DOWN, UP)]
            for substates2 in [(UP, DOWN), (DOWN, UP)]
                # prefactor from the - in s_y
                c = substates1 == substates2 ? +1.0 : -1.0
                sign1, _state = annihilate(state, site2, substates2[1])
                sign2, _state = create(_state, site2, substates2[2])
                sign3, _state = annihilate(_state, site1, substates1[1])
                sign4, _state = create(_state, site1, substates1[2])
                if _state != 0
                    push!(prefactors, -1.0 * c *  sign1 * sign2 * sign3 * sign4)
                    push!(states, _state)
                end
            end
        end
        states, prefactors
    end
end
function spin_density_correlation_z(site1, site2)
    state -> begin
        states = typeof(state)[]
        prefactors = Float64[]
        for substates1 in [(UP, UP), (DOWN, DOWN)]
            for substates2 in [(UP, UP), (DOWN, DOWN)]
                # prefactor from the - in s_z
                c = substates1 == substates2 ? +1.0 : -1.0
                sign1, _state = annihilate(state, site2, substates2[1])
                sign2, _state = create(_state, site2, substates2[2])
                sign3, _state = annihilate(_state, site1, substates1[1])
                sign4, _state = create(_state, site1, substates1[2])
                if _state != 0
                    push!(prefactors, c * sign1 * sign2 * sign3 * sign4)
                    push!(states, _state)
                end
            end
        end
        states, prefactors
    end
end


function pairing_correlation(site1, site2)
    state -> begin
        sign1, _state = create(state, site2, DOWN)
        sign2, _state = create(_state, site2, UP)
        sign3, _state = annihilate(_state, site1, DOWN)
        sign4, _state = annihilate(_state, site1, UP)
        if _state == 0
            return typeof(state)[], Float64[]
        else
            return [_state], [-sign1 * sign2 * sign3 * sign4]
        end
    end
end


function expectation_value(
        observable::Function,
        H;
        T=1.0, beta = 1.0 / T,
        N_sites = 4,
        N_substates = 2
    )

    vals, vecs = eigen(H)
    Z = 0.0
    O = 0.0
    for i in eachindex(vals)
        # exp(βEᵢ)
        temp = exp(-beta * vals[i])
        Z += temp

        # ⟨ψᵢ|Ô|ψᵢ⟩
        T = eltype(vecs)
        right_coefficients = zeros(T <: Complex ? T : Complex{T}, size(vecs, 1))
        for j in 1:size(vecs, 1)
            state = state_from_integer(j-1, N_sites, N_substates)
            states, values = observable(state)
            for (s, v) in zip(states, values)
                # Assuming no (s, v) pair if state destroyed
                k = s.chunks[1]+1
                right_coefficients[k] += v * vecs[j, i]
            end
        end
        O += temp * dot(vecs[:, i], right_coefficients)
    end
    O / Z
end


function calculate_Greens_matrix(H, lattice; beta=1.0, N_substates=2)
    G = Matrix{ComplexF64}(
        undef,
        lattice.sites*N_substates,
        lattice.sites*N_substates
    )
    for substate1 in 1:N_substates, substate2 in 1:N_substates
        for site1 in 1:lattice.sites, site2 in 1:lattice.sites
            G[
                lattice.sites * (substate1-1) + site1,
                lattice.sites * (substate2-1) + site2
            ] = expectation_value(
                Greens(site1, site2, substate1, substate2),
                H,
                beta = beta,
                N_sites = lattice.sites,
                N_substates=N_substates
            )
        end
    end
    G
end



# utility

function state2string(state)
    sub, L = size(state)
    spin = ["↑", "↓"]
    chunks = String[]
    for p in 1:L
        str = ""
        for s in 1:2
            str *= state[s, p] ? spin[s] : "⋅"
        end
        push!(chunks, str)
    end
    join(chunks, " ")
end
