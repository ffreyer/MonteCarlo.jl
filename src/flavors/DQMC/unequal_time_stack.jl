mutable struct UnequalTimeStack{GT<:Number, GMT<:AbstractArray{GT}} <: AbstractDQMCStack
    # B_{n*safe_mult} ... B_1
    forward_idx::Int64
    forward_u_stack::Vector{GMT}
    forward_d_stack::Vector{Vector{Float64}}
    forward_t_stack::Vector{GMT}
    
    # B_{n*safe_mult+1}^† ... B_M^†
    backward_idx::Int64
    backward_u_stack::Vector{GMT}
    backward_d_stack::Vector{Vector{Float64}}
    backward_t_stack::Vector{GMT}
    
    # B_{n*safe_mult+1}^-1 .. B_{(n+1)*safe_mult}^-1
    inv_done::Vector{Bool}
    inv_u_stack::Vector{GMT}
    inv_d_stack::Vector{Vector{Float64}}
    inv_t_stack::Vector{GMT}

    # Greens construction
    greens::GMT
    tmp::GMT
    U::GMT
    D::Vector{Float64}
    T::GMT

    # To avoid recalculating
    last_update::Int64
    last_slices::Tuple{Int64, Int64}

    function UnequalTimeStack{GT, GMT}() where {GT<:Number, GMT<:AbstractArray{GT}}
        @assert isconcretetype(GT);
        @assert isconcretetype(GMT);
        new{GT, GMT}()
    end
end

# TODO
# function Base.sizeof(::Type{UnequalTimeStack}, mc::DQMC)
#     GreensEltype =  MonteCarlo.geltype(mc)
#     N = length(mc.model.l)
#     flv = mc.model.flv
#     blocks = mc.s.n_elements

#     sizeof(GreensEltype) * flv*N * flv*N * blocks * 4 +         # forward+backward U, T
#     sizeof(GreensEltype) * flv*N * blocks * 2 +                 # forward+backward D
#     sizeof(GreensEltype) * flv*N * flv*N * (blocks-1) * 2 +     # inv U, T
#     sizeof(GreensEltype) * flv*N * (blocks-1) +                 # inv Dr
#     5*8 + sizeof(Bool) * (blocks-1) +                           # book-keeping/skipping
#     sizeof(GreensEltype) * flv*N * flv*N * 3 +                  # greens, U, Tr
#     sizeof(GreensEltype) * flv*N                                # D
# end

function initialize_stack(mc::DQMC, s::UnequalTimeStack)
    GreensElType  = geltype(mc)
    GreensMatType = gmattype(mc)
    N = length(lattice(mc))
    flv = nflavors(mc.model)
    M = convert(Int, mc.p.slices / mc.p.safe_mult) + 1

    # B_{n*safe_mult} ... B_1
    s.forward_idx = 1
    s.forward_u_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:M]
    s.forward_d_stack = [zeros(Float64, flv*N) for _ in 1:M]
    s.forward_t_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:M]
    
    # B_{n*safe_mult+1}^† ... B_M^†
    s.backward_idx = M-1
    s.backward_u_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:M]
    s.backward_d_stack = [zeros(Float64, flv*N) for _ in 1:M]
    s.backward_t_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:M]
    
    # B_{n*safe_mult+1}^-1 .. B_{(n+1)*safe_mult}^-1
    s.inv_done = fill(false, M-1)
    s.inv_u_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:M-1]
    s.inv_d_stack = [zeros(Float64, flv*N) for _ in 1:M-1]
    s.inv_t_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:M-1]

    # Greens construction
    s.greens = GreensMatType(undef, flv*N, flv*N)
    s.tmp = GreensMatType(undef, flv*N, flv*N)
    s.U = GreensMatType(undef, flv*N, flv*N)
    s.D = zeros(Float64, flv*N)
    s.T = GreensMatType(undef, flv*N, flv*N)

    # To avoid recalculating
    last_update = -1
    last_slices = (-1, -1)

    # maybe skip identities?
    copyto!(s.forward_u_stack[1], I)
    s.forward_d_stack[1] .= 1
    copyto!(s.forward_t_stack[1], I)

    copyto!(s.backward_u_stack[end], I)
    s.backward_d_stack[end] .= 1
    copyto!(s.backward_t_stack[end], I)

    s
end

########################################
# Stack building (full & lazy)
########################################

@bm function build_stack(mc::DQMC, s::UnequalTimeStack)
    # stack = [0, Δτ, 2Δτ, ..., β]
    if mc.last_sweep == s.last_update && all(s.inv_done) && 
        (s.forward_idx == length(mc.s.ranges)+1) && (s.backward_idx == 1)
        return nothing
    end

    # forward
    @bm "forward build" begin
        @inbounds for idx in 1:length(mc.s.ranges)
            copyto!(mc.s.curr_U, s.forward_u_stack[idx])
            for slice in mc.s.ranges[idx]
                multiply_slice_matrix_left!(mc, mc.model, slice, mc.s.curr_U)
            end
            vmul!(mc.s.tmp1, mc.s.curr_U, Diagonal(s.forward_d_stack[idx]))
            udt_AVX_pivot!(
                s.forward_u_stack[idx+1], s.forward_d_stack[idx+1], 
                mc.s.tmp1, mc.s.pivot, mc.s.tempv
            )
            vmul!(s.forward_t_stack[idx+1], mc.s.tmp1, s.forward_t_stack[idx])
        end
    end

    # backward
    @bm "backward build" begin
        @inbounds for idx in length(mc.s.ranges):-1:1
            copyto!(mc.s.curr_U, s.backward_u_stack[idx + 1])
            for slice in reverse(mc.s.ranges[idx])
                multiply_daggered_slice_matrix_left!(mc, mc.model, slice, mc.s.curr_U)
            end
            vmul!(mc.s.tmp1, mc.s.curr_U, Diagonal(s.backward_d_stack[idx + 1]))
            udt_AVX_pivot!(
                s.backward_u_stack[idx], s.backward_d_stack[idx], 
                mc.s.tmp1, mc.s.pivot, mc.s.tempv
            )
            vmul!(s.backward_t_stack[idx], mc.s.tmp1, s.backward_t_stack[idx+1])
        end
    end

    # inverse
    # TODO should this multiply to U's instead?
    @bm "inverse build" begin
        @inbounds for idx in 1:length(mc.s.ranges)
            copyto!(s.inv_t_stack[idx], I)
            for slice in reverse(mc.s.ranges[idx])
                multiply_slice_matrix_inv_left!(mc, mc.model, slice, s.inv_t_stack[idx])
            end
            udt_AVX_pivot!(
                s.inv_u_stack[idx], s.inv_d_stack[idx], s.inv_t_stack[idx], 
                mc.s.pivot, mc.s.tempv
            )
        end
    end

    s.inv_done .= true
    s.forward_idx = length(mc.s.ranges)+1
    s.backward_idx = 1
    s.last_update = mc.last_sweep

    nothing
end

@bm function lazy_build_forward!(mc::DQMC, s::UnequalTimeStack, upto)
    # stack = [0, Δτ, 2Δτ, ..., β]
    if s.last_update != mc.last_sweep
        s.last_update = mc.last_sweep
        s.inv_done .= false
        s.forward_idx = 1
        s.backward_idx = length(mc.s.ranges)+1
    end

    # forward
    @bm "forward build" begin
        @inbounds for idx in s.forward_idx:upto-1
            copyto!(mc.s.curr_U, s.forward_u_stack[idx])
            for slice in mc.s.ranges[idx]
                multiply_slice_matrix_left!(mc, mc.model, slice, mc.s.curr_U)
            end
            vmul!(mc.s.tmp1, mc.s.curr_U, Diagonal(s.forward_d_stack[idx]))
            udt_AVX_pivot!(
                s.forward_u_stack[idx+1], s.forward_d_stack[idx+1], 
                mc.s.tmp1, mc.s.pivot, mc.s.tempv
            )
            vmul!(s.forward_t_stack[idx+1], mc.s.tmp1, s.forward_t_stack[idx])
        end
    end

    s.forward_idx = max(upto, s.forward_idx)
    nothing
end

@bm function lazy_build_backward!(mc::DQMC, s::UnequalTimeStack, downto)
    if s.last_update != mc.last_sweep
        s.last_update = mc.last_sweep
        s.inv_done .= false
        s.forward_idx = 1
        s.backward_idx = length(mc.s.ranges)+1
    end

    # backward
    @bm "backward build" begin
        @inbounds for idx in s.backward_idx-1:-1:downto
            copyto!(mc.s.curr_U, s.backward_u_stack[idx + 1])
            for slice in reverse(mc.s.ranges[idx])
                multiply_daggered_slice_matrix_left!(mc, mc.model, slice, mc.s.curr_U)
            end
            vmul!(mc.s.tmp1, mc.s.curr_U, Diagonal(s.backward_d_stack[idx + 1]))
            udt_AVX_pivot!(
                s.backward_u_stack[idx], s.backward_d_stack[idx], 
                mc.s.tmp1, mc.s.pivot, mc.s.tempv
            )
            vmul!(s.backward_t_stack[idx], mc.s.tmp1, s.backward_t_stack[idx+1])
        end
    end

    s.backward_idx = min(downto, s.backward_idx)
end

@bm function lazy_build_inv!(mc::DQMC, s::UnequalTimeStack, from, to)
    if s.last_update != mc.last_sweep
        s.last_update = mc.last_sweep
        s.inv_done .= false
        s.forward_idx = 1
        s.backward_idx = length(mc.s.ranges)+1
    end

    # inverse
    @bm "inverse build" begin
        @inbounds for idx in from:to
            s.inv_done[idx] && continue
            s.inv_done[idx] = true
            copyto!(s.inv_t_stack[idx], I)
            for slice in reverse(mc.s.ranges[idx])
                multiply_slice_matrix_inv_left!(mc, mc.model, slice, s.inv_t_stack[idx])
            end
            udt_AVX_pivot!(
                s.inv_u_stack[idx], s.inv_d_stack[idx], s.inv_t_stack[idx], 
                mc.s.pivot, mc.s.tempv
            )
        end
    end

    nothing
end



########################################
# Calculating greens
########################################



"""
    greens(mc::DQMC, k, l)

Calculates the unequal-time Greens function at slice `k` and `l`, i.e.
`G(k <- l) = G(kΔτ <- lΔτ) = ⟨cᵢ(k⋅Δτ)cⱼ(l⋅Δτ)^†⟩` where `nslices(mc) ≥ k > l ≥ 0`.

Note that `G(0, 0) = G(nslices, nslices) = G(β, β) = G(0, 0)`.

This method requires the `UnequalTimeStack` to be available. The stack 
variables `Ul`, `Dl`, `Tl`, `Ur`, `Dr`, `Tr`, `curr_U` and `tmp1` will
be overwritten (but are avaible as `output` and `temp`).
"""
@bm greens(mc::DQMC, slice1::Int64, slice2::Int64) = copy(_greens!(mc, slice1, slice2))
"""
    greens!(mc::DQMC, k, l[; output=mc.s.greens_temp, temp=mc.s.tmp1])
"""
@bm function greens!(
        mc::DQMC, slice1::Int64, slice2::Int64; 
        output = mc.s.greens_temp, temp = mc.s.tmp1
    ) 
    _greens!(mc, slice1, slice2, output, temp)
end
"""
    _greens!(mc::DQMC, k, l, output=mc.s.greens_temp, temp=mc.s.tmp1)
"""
function _greens!(
        mc::DQMC, slice1::Int64, slice2::Int64,
        output = mc.s.greens_temp, temp = mc.s.tmp1
    )
    calculate_greens(mc, slice1, slice2)
    return _greens!(mc, output, mc.ut_stack.greens, temp)
end
@bm function calculate_greens(mc::DQMC, slice1::Int64, slice2::Int64)
    @assert 0 ≤ slice1 ≤ mc.p.slices
    @assert 0 ≤ slice2 ≤ mc.p.slices
    s = mc.ut_stack
    if (s.last_slices != (slice1, slice2)) || (s.last_update != mc.last_sweep)
        s.last_slices = (slice1, slice2)
        if slice1 ≥ slice2
            calculate_greens_full1!(mc, s, slice1, slice2)
        else
            calculate_greens_full2!(mc, s, slice1, slice2)
        end
    end
    s.greens
end








# low = slice2, high = slice1
"""
    compute_inverse_udt_block!(mc, s::UnequalTimeStack, low, high[, ...])

Computes a inverse UDT block `U D T = B_{low+1}^-1 B_{low+2}^-1 ⋯ B_high^-1`.

Default extra args are: (in order)
* `U = s.U`
* `D = s.D`
* `T = s.T`
* `tmp1 = mc.s.tmp1`
* `tmp2 = mc.s.tmp2`
"""
function compute_inverse_udt_block!(
        mc, s, low, high,
        U = s.U, D = s.D, T = s.T, tmp1 = mc.s.tmp1, tmp2 = mc.s.tmp2
    )
    # @assert low ≤ high
    # B_{low+1}^-1 B_{low+2}^-1 ⋯ B_{high-1}^-1 B_high^-1

    # Combine pre-computed blocks
    @bm "inverse pre-computed" begin
        lower = div(low+1 + mc.p.safe_mult - 2, mc.p.safe_mult) + 1
        upper = div(high, mc.p.safe_mult)
        lazy_build_inv!(mc, s, lower, upper)
        copyto!(U, I)
        D .= 1
        copyto!(T, I)
        # UDT combining style
        for idx in lower:upper
            # () is operation order, [] marks UDT decomposition
            # U [(D (T stack_U)) stack_D] stack_T
            # (U [U') D (T'] stack_T)
            # U D T
            vmul!(tmp1, T, s.inv_u_stack[idx])
            vmul!(tmp2, Diagonal(D), tmp1)
            vmul!(tmp1, tmp2, Diagonal(s.inv_d_stack[idx]))
            udt_AVX_pivot!(tmp2, D, tmp1, mc.s.pivot, mc.s.tempv)
            vmul!(T, tmp1, s.inv_t_stack[idx])
            vmul!(tmp1, U, tmp2)
            copyto!(U, tmp1)
        end
    end

    # remaining multiplications to reach specified bounds
    @bm "inverse fine tuning" begin
        lower_slice = (lower-1) * mc.p.safe_mult + 1
        upper_slice = upper * mc.p.safe_mult

        for slice in min(lower_slice-1, high) : -1 : low+1
            multiply_slice_matrix_inv_left!(mc, mc.model, slice, U)
        end
        if min(lower_slice-1, high) ≥ low+1
            vmul!(tmp1, U, Diagonal(D))
            udt_AVX_pivot!(U, D, tmp1, mc.s.pivot, mc.s.tempv)
            vmul!(tmp2, tmp1, T)
            copyto!(T, tmp2)
        end
        for slice in max(upper_slice+1, min(lower_slice-1, high)+1) : high
            multiply_slice_matrix_inv_right!(mc, mc.model, slice, T)
        end
        # U D T = B_{l+1}^-1 B_{l+2}^-1 ⋯ B_{k-1}^-1 B_k^-1
    end
    return nothing
end


"""
    compute_forward_udt_block!(mc, s::UnequalTimeStack, slice[, ...])

Computes a UDT block `U D T = B_slice B_{slice-1} ⋯ B_1`.

Default extra args are: (in order)
* `U = mc.s.Ul`
* `D = mc.s.Dl`
* `T = mc.s.Tl`
* `tmp = mc.s.tmp1`
"""
function compute_forward_udt_block!(
        mc, s, slice, apply_pivot = Val(true),
        U = mc.s.Ul, D = mc.s.Dl, T = mc.s.Tl, tmp = mc.s.tmp1,
    )
    # B(slice, 1) = B_slice B_{slice-1} ⋯ B_1 = Ul Dl Tl
    @bm "forward B" begin
        idx = div(slice-1, mc.p.safe_mult)
        lazy_build_forward!(mc, s, idx+1)
        copyto!(T, s.forward_u_stack[idx+1])
        for l in mc.p.safe_mult * idx + 1 : slice
            multiply_slice_matrix_left!(mc, mc.model, l, T)
        end
        vmul!(tmp, T, Diagonal(s.forward_d_stack[idx+1]))
        udt_AVX_pivot!(U, D, tmp, mc.s.pivot, mc.s.tempv, apply_pivot)
        vmul!(T, tmp, s.forward_t_stack[idx+1])
    end
    return nothing
end


"""
    compute_backward_udt_block!(mc, s::UnequalTimeStack, slice[, ...])

Computes a UDT block `(U D T)^† = T^† D U^† = B_N B_{N-1} ⋯ B_slice`.

Default extra args are: (in order)
* `U = mc.s.Ur`
* `D = mc.s.Dr`
* `T = mc.s.Tr`
* `tmp = mc.s.tmp1`
"""
function compute_backward_udt_block!(
        mc, s, slice,
        U = mc.s.Ur, D = mc.s.Dr, T = mc.s.Tr, tmp = mc.s.tmp1
    )
    # B(N, slice) = B_N B_{N-1} ⋯ B_{slice+1} = (Ur Dr Tr)^† = Tr^† Dr^† Ur^†
    @bm "backward B" begin
        idx = div.(slice + mc.p.safe_mult - 1, mc.p.safe_mult)
        lazy_build_backward!(mc, s, idx+1)
        copyto!(U, s.backward_u_stack[idx+1])
        for l in mc.p.safe_mult * idx : -1 : slice+1
            multiply_daggered_slice_matrix_left!(mc, mc.model, l, U)
        end
        vmul!(tmp, U, Diagonal(s.backward_d_stack[idx+1]))
        udt_AVX_pivot!(U, D, tmp, mc.s.pivot, mc.s.tempv)
        vmul!(T, tmp, s.backward_t_stack[idx + 1])
    end
    return nothing
end



@bm function calculate_greens_full1!(mc, s, slice1, slice2)
    # stack = [0, Δτ, 2Δτ, ..., β] = [0, safe_mult, 2safe_mult, ... N]
    # @assert slice1 ≥ slice2

    # k ≥ l or slice1 ≥ slice2
    # B_{l+1}^-1 B_{l+2}^-1 ⋯ B_{k-1}^-1 B_k^-1
    compute_inverse_udt_block!(mc, s, slice2, slice1) # low high

    # B(slice2, 1) = Ul Dl Tl
    compute_forward_udt_block!(mc, s, slice2)

    # B(N, slice1) = (Ur Dr Tr)^† = Tr^† Dr^† Ur^†
    compute_backward_udt_block!(mc, s, slice1)



    # U D T remains valid here
    @bm "compute G" begin
        # [B_{l+1}^-1 B_{l+2}^-1 ⋯ B_k^-1 + B_l ⋯ B_1 B_N ⋯ B_{k+1}]^-1
        # [U D T + Ul (Dl Tl Tr^† Dr) Ur^†]^-1
        @bm "B1" begin
            vmul!(s.greens, mc.s.Tl, adjoint(mc.s.Tr))
            vmul!(mc.s.tmp1, s.greens, Diagonal(mc.s.Dr))
            vmul!(s.greens, Diagonal(mc.s.Dl), mc.s.tmp1)
        end
        # [U D T + Ul (G) Ur^†]^-1
        @bm "udt" begin
            udt_AVX_pivot!(mc.s.Tr, mc.s.Dr, s.greens, mc.s.pivot, mc.s.tempv, Val(false))
        end
        # [U D T + (Ul Tr) Dr (G Ur^†)]^-1
        @bm "B2" begin
            vmul!(mc.s.Tl, mc.s.Ul, mc.s.Tr)
            # (G Ur^†) = (Ur / G)^-1
            # Ur := Ur / G
            rdivp!(mc.s.Ur, s.greens, mc.s.Ul, mc.s.pivot) 
        end
        # [U D T + Tl Dr Ur^-1]^-1
        # [U (D T Ur + U^† Tl Dr) Ur^-1]^-1
        # [U D_max (D_min T Ur 1/Dr_max + 1/D_max U^† Tl Dr_min) Dr_max Ur^-1]^-1
        @bm "B3" begin
            # 1/D_max U^† Tl Dr_min
            vmul!(mc.s.Tr, adjoint(s.U), mc.s.Tl)
            vmaxinv!(mc.s.Dl, s.D) # mc.s.Dl .= 1.0 ./ max.(1.0, s.D)
            vmul!(mc.s.tmp1, Diagonal(mc.s.Dl), mc.s.Tr)
            vmin!(mc.s.Dl, mc.s.Dr) # mc.s.Dl .= min.(1.0, mc.s.Dr)
            vmul!(mc.s.Tr, mc.s.tmp1, Diagonal(mc.s.Dl))
        end
        # [U D_max (D_min T Ur 1/Dr_max + Tr) Dr_max Ur^-1]^-1
        @bm "B4" begin
            # D_min T Ur 1/Dr_max
            vmul!(mc.s.Tl, s.T, mc.s.Ur)
            vmin!(mc.s.Dl, s.D) # mc.s.Dl .= min.(1.0, s.D)
            vmul!(mc.s.tmp1, Diagonal(mc.s.Dl), mc.s.Tl)
            vmaxinv!(mc.s.Dl, mc.s.Dr) # mc.s.Dl .= 1.0 ./ max.(1.0, mc.s.Dr)
            vmul!(mc.s.Tl, mc.s.tmp1, Diagonal(mc.s.Dl))
        end
        # [U D_max (Tl + Tr) Dr_max Ur^-1]^-1
        @bm "sum, UDT" begin
            rvadd!(mc.s.Tl, mc.s.Tr)
            udt_AVX_pivot!(mc.s.Tr, mc.s.Dl, mc.s.Tl, mc.s.pivot, mc.s.tempv, Val(false))
        end
        # [U D_max (Tr Dl Tl) Dr_max Ur^-1]^-1
        # Ur 1/Dr_max Tl^-1 1/Dl Tr^† D_max U^†
        @bm "B5" begin
            # [[((1/Dr_max) / Tl) 1/Dl] Tr^†] D_max
            vmaxinv!(mc.s.Dr, mc.s.Dr) # mc.s.Dr .= 1.0 ./ max.(1.0, mc.s.Dr)
            copyto!(mc.s.Ul, Diagonal(mc.s.Dr))
            rdivp!(mc.s.Ul, mc.s.Tl, mc.s.tmp1, mc.s.pivot)
            vinv!(mc.s.Dl) # mc.s.Dl[i] .= 1.0 ./ mc.s.Dl[i]
            vmul!(mc.s.tmp1, mc.s.Ul, Diagonal(mc.s.Dl))
            vmul!(mc.s.Ul, mc.s.tmp1, adjoint(mc.s.Tr))
            vmaxinv!(mc.s.Dl, s.D) # mc.s.Dl .= 1.0 ./ max.(1.0, s.D)
            vmul!(s.greens, mc.s.Ul, Diagonal(mc.s.Dl))
        end
        # Ur G U^†
        @bm "B6" begin
            vmul!(mc.s.Tr, s.greens, adjoint(s.U))
            vmul!(s.greens, mc.s.Ur, mc.s.Tr)
        end
    end

    s.greens
end



@bm function calculate_greens_full2!(mc, s, slice1, slice2)
    # stack = [0, Δτ, 2Δτ, ..., β] = [0, safe_mult, 2safe_mult, ... N]
    # @assert slice1 ≤ slice2

    # k ≤ l or slice1 ≤ slice2
    # B_{k+1}^-1 B_{k+2}^-1 ⋯ B_{l-1}^-1 B_l^-1
    compute_inverse_udt_block!(mc, s, slice1, slice2) # low high

    # B(slice1, 1) = B_k ⋯ B_1 = Ul Dl Tl
    compute_forward_udt_block!(mc, s, slice1)
    
    # B(M, slice2) = B_M ⋯ B_slice2+1 = (Ur Dr Tr)^† = Tr^† Dr^† Ur^†
    compute_backward_udt_block!(mc, s, slice2)


    # U D T remains valid here
    @bm "compute G" begin
        # [B_{l} B_{l-1} ⋯ B_{k+1} + (B_k ⋯ B_1 B_M ⋯ B_{l+1})^-1]^-1
        # [T^-1 D^-1 U^† + (Ul Dl Tl Tr^† Dr Ur^†)^-1]^-1
        # [T^-1 D^-1 U^† + Ur (Dl Tl Tr^† Dr)^-1 Ul^†]^-1
        @bm "B1" begin
            vmul!(s.greens, mc.s.Tl, adjoint(mc.s.Tr))
            vmul!(mc.s.tmp1, Diagonal(mc.s.Dl), s.greens)
            vmul!(s.greens, mc.s.tmp1, Diagonal(mc.s.Dr))
        end
        # [T^-1 D^-1 U^† + Ur G^-1 Ul^†]^-1
        @bm "udt" begin
            udt_AVX_pivot!(mc.s.Tr, mc.s.Dr, s.greens, mc.s.pivot, mc.s.tempv, Val(false))
        end
        # [T^-1 D^-1 U^† + Ur G^-1 Dr^-1 Tr^† Ul^†]^-1
        # [T^-1 D_min^-1 (D_max^-1 U^† (Ul Tr) Dr_min + D_min T Ur G^-1 Dr_max^-1) Dr_min^-1 (Ul Tr)^†]^-1
        @bm "B2" begin
            # D_max^-1 U^† (Ul Tr) Dr_min
            vmul!(mc.s.Tl, mc.s.Ul, mc.s.Tr) # keep me alive
            vmul!(mc.s.Ul, adjoint(s.U), mc.s.Tl)
            vmaxinv!(mc.s.Dl, s.D)
            vmul!(s.U, Diagonal(mc.s.Dl), mc.s.Ul)
            vmin!(mc.s.Dl, mc.s.Dr)
            vmul!(mc.s.Ul, s.U, Diagonal(mc.s.Dl))
        end
        # [T^-1 D_min^-1 (Ul + D_min T Ur G^-1 Dr_max^-1) Dr_min^-1 Tl^†]^-1
        @bm "B3" begin
            # D_min T Ur G^-1 Dr_max^-1
            vmul!(s.U, s.T, mc.s.Ur)
            rdivp!(s.U, s.greens, mc.s.Ur, mc.s.pivot)
            vmin!(mc.s.Dl, s.D)
            vmul!(mc.s.Ur, Diagonal(mc.s.Dl), s.U)
            vmaxinv!(mc.s.Dl, mc.s.Dr)
            vmul!(mc.s.Tr, mc.s.Ur, Diagonal(mc.s.Dl))
        end
        # [T^-1 D_min^-1 (Ul + Tr) Dr_min^-1 Tl^†]^-1
        @bm "sum, udt" begin
            rvadd!(mc.s.Tr, mc.s.Ul)
            udt_AVX_pivot!(mc.s.Ul, mc.s.Dl, mc.s.Tr, mc.s.pivot, mc.s.tempv, Val(false))
        end
        # [T^-1 D_min^-1 Ul Dl Tr Dr_min^-1 Tl^-1]^-1
        # Tl ({[(Dr_min / Tr) Dl^-1] Ul^†} D_min) T
        @bm "B4" begin
            vmin!(mc.s.Dr, mc.s.Dr)
            copyto!(s.U, Diagonal(mc.s.Dr))
            rdivp!(s.U, mc.s.Tr, mc.s.Ur, mc.s.pivot)
            vinv!(mc.s.Dl)
            vmul!(mc.s.Ur, s.U, Diagonal(mc.s.Dl))
            vmul!(s.U, mc.s.Ur, adjoint(mc.s.Ul))
            vmin!(s.D, s.D)
            vmul!(mc.s.Ur, s.U, Diagonal(s.D))
        end
        # Tl Ur T
        @bm "B6" begin
            vmul!(mc.s.Tr, mc.s.Ur, s.T)
            vmul!(s.greens, mc.s.Tl, mc.s.Tr)
            rmul!(s.greens, -1.0)
        end
    end

    s.greens
end



################################################################################
### Iterators
################################################################################



# The method above provides the unequal-time Greens function in a stable, 
# general but expensive way. The idea of iterators is to provide what is needed
# more quickly, without the UnequalTimeStack and without having to worry about
# managing stack variables.

# Currently we have:
# GreensIterator
#   - attempts to be general (but isn't yet)
#   - requires UnequalTimeStack
#   - should have high accuracy, but still faster than manual
# CombinedGreensIterator
#   - does just l = 0, k = 0:M-1, but returns G_kl and G_ll
#   - does not require UnequalTimeStack
#   - same accuracy as GreensIterator

abstract type AbstractGreensIterator end
abstract type AbstractUnequalTimeGreensIterator <: AbstractGreensIterator end

init!(::AbstractUnequalTimeGreensIterator) = nothing
"""
    verify(iterator::AbstractUnequalTimeGreensIterator[, maxerror=1e-6])

Returns true if the given `iterator` is accurate up to the given `maxerror`.

See also: (@ref)[`accuracy`]
"""
verify(it::AbstractUnequalTimeGreensIterator, maxerror=1e-6) = maximum(accuracy(it)) < maxerror

# Maybe split into multiple types?
struct GreensIterator{slice1, slice2, T <: DQMC} <: AbstractUnequalTimeGreensIterator
    mc::T
    recalculate::Int64
end

"""
    GreensIterator(mc::DQMC[, ks=Colon(), ls=0, recalculate=4mc.p.safe_mult])

Creates an Iterator which calculates all `[G[k <- l] for l in ls for k in ks]`
efficiently. Currently only allows `ks = :, ls::Integer`. 

For stability it is necessary to fully recompute `G[k <- l]` every so often. one
can adjust the frequency of recalculations via `recalculate`. To estimate the 
resulting accuracy, one may use `accuracy(GreensIterator(...))`.

This iterator requires the `UnequalTimeStack`. Iteration overwrites the 
`DQMCStack` variables `curr_U`, `Ul`, `Dl`, `Tl`, `Ur`, `Dr`, `Tr`, `tmp1` and 
`tmp2`, with `curr_U` acting as the output. For the iteration to remain valid, 
the UnequalTimeStack must not be overwritten. As such:
- `greens!(mc)` can be called and remains valid
- `greens!(mc, slice)` can be called and remains valid
- `greens!(mc, k, l)` will break iteration but remains valid (call before iterating)
"""
function GreensIterator(
        mc::T, slice1 = Colon(), slice2 = 0, recalculate = 4mc.p.safe_mult
    ) where {T <: DQMC}
    GreensIterator{slice1, slice2, T}(mc, recalculate)
end
init!(it::GreensIterator) = initialize_stack(it.mc, it.mc.ut_stack)
Base.length(it::GreensIterator{:, i}) where {i} = it.mc.p.slices + 1 - i

# Slower, versatile version:
function Base.iterate(it::GreensIterator{:, i}) where {i}
    s = it.mc.ut_stack
    calculate_greens_full1!(it.mc, s, i, i)
    copyto!(s.T, s.greens)
    udt_AVX_pivot!(s.U, s.D, s.T, it.mc.s.pivot, it.mc.s.tempv)
    G = _greens!(it.mc, it.mc.s.curr_U, s.greens, it.mc.s.Ur)
    return (G, (i+1, i))
end
function Base.iterate(it::GreensIterator{:}, state)
    s = it.mc.ut_stack
    k, l = state
    if k > it.mc.p.slices
        return nothing
    elseif k % it.recalculate == 0
        # Recalculate
        calculate_greens_full1!(it.mc, s, k, l) # writes s.greens
        G = _greens!(it.mc, it.mc.s.curr_U, s.greens, it.mc.s.Tl)
        copyto!(s.T, s.greens)
        udt_AVX_pivot!(s.U, s.D, s.T, it.mc.s.pivot, it.mc.s.tempv)
        return (G, (k+1, l))
    elseif k % it.mc.p.safe_mult == 0
        # Stabilization
        multiply_slice_matrix_left!(it.mc, it.mc.model, k, s.U)
        vmul!(it.mc.s.curr_U, s.U, Diagonal(s.D))
        vmul!(it.mc.s.tmp1, it.mc.s.curr_U, s.T)
        udt_AVX_pivot!(s.U, s.D, it.mc.s.curr_U, it.mc.s.pivot, it.mc.s.tempv)
        vmul!(it.mc.s.tmp2, it.mc.s.curr_U, s.T)
        copyto!(s.T, it.mc.s.tmp2)
        G = _greens!(it.mc, it.mc.s.curr_U, it.mc.s.tmp1, it.mc.s.tmp2)
        return (G, (k+1, l))
    else
        # Quick advance
        multiply_slice_matrix_left!(it.mc, it.mc.model, k, s.U)
        vmul!(it.mc.s.curr_U, s.U, Diagonal(s.D))
        vmul!(it.mc.s.tmp1, it.mc.s.curr_U, s.T)
        G = _greens!(it.mc, it.mc.s.curr_U, it.mc.s.tmp1, it.mc.s.Ur)
        s.last_slices = (-1, -1) # for safety
        return (G, (k+1, l))
    end
end
"""
    accuracy(iterator::AbstractUnequalTimeGreensIterator)

Compares values from the given iterator to more verbose computations, returning 
the maximum differences for each. This can be used to check numerical stability.
"""
function accuracy(iter::GreensIterator{:, l}) where {l}
    mc = iter.mc
    Gk0s = [deepcopy(greens(mc, k, l)) for k in l:nslices(mc)]
    [maximum(abs.(Gk0s[i] .- G)) for (i, G) in enumerate(iter)]
end



"""
    CombinedGreensIterator(mc::DQMC, recalculate=4mc.p.safe_mult)

Returns an iterator which iterates `[(G[k, 0], G[k, k]) for k in 0:nslices-1]`. 
Does a full recalculation of `G[k, 0]` and `G[k, k]` if `k % recalculate == 0`.


This iterator uses `s.tmp1` and `s.tmp2` from the normal `s::DQMCSTack` and 
`uts.greens` from the `uts::UnequalTimeStack` as outputs. For correct iteration 
it requires  `s.Ul`, `s.Dl`, `s.Tl`, `s.Ur`, `s.Dr`, `s.Tr`, `uts.U`, `uts.D`
and `uts.T` to remain unchanged. Further `s.curr_U` and `uts.tmp` are 
overwritten. As such
- `greens!(mc)` will break iteration but remains valid (call before iterating)
- `greens!(mc, slice)` will break iteration but remains valid
- `greens!(mc, k, l)` will break iteration but remains valid
"""
struct CombinedGreensIterator{T <: DQMC} <: AbstractUnequalTimeGreensIterator
    mc::T
    recalculate::Int64
end

function CombinedGreensIterator(mc::T, model::Model, recalculate::Int64=4mc.p.safe_mult) where T
    CombinedGreensIterator{T}(mc, recalculate)
end
function CombinedGreensIterator(mc::T, recalculate::Int64=4mc.p.safe_mult) where T
    CombinedGreensIterator{T}(mc, recalculate)
end
function init!(it::CombinedGreensIterator)
    if it.recalculate < nslices(it.mc)
        initialize_stack(it.mc, it.mcm.ut_stack)
    end
end

Base.length(it::CombinedGreensIterator) = nslices(it.mc)


# Fast specialized version
function Base.iterate(it::CombinedGreensIterator)
    # Measurements take place at current_slice = 1 <> l = τ = 0
    @assert current_slice(it.mc) == 1
    s = it.mc.s
    uts = it.mc.ut_stack
    copyto!(s.Tl, s.greens)
    # Need full built stack for curr_U to be save
    build_stack(it.mc, uts)

    udt_AVX_pivot!(s.Ul, s.Dl, s.Tl, it.mc.s.pivot, it.mc.s.tempv)
    copyto!(uts.U, s.Ul)
    copyto!(uts.D, s.Dl)
    copyto!(uts.T, s.Tl)

    # G01 = (G00-I)B_1^-1; then G0l+1 = G0l B_{l+1}^-1
    vsub!(s.Tr, s.greens, I)
    udt_AVX_pivot!(s.Ur, s.Dr, s.Tr, it.mc.s.pivot, it.mc.s.tempv)
    return iterate(it, 1)
end
# probably need extra temp variables
function Base.iterate(it::CombinedGreensIterator, l)
    # l is 1-based
    s = it.mc.s
    uts = it.mc.ut_stack
    if l > it.mc.p.slices
        return nothing
    elseif l % it.recalculate == 0 
        # Recalculation will overwrite 
        # Ul, Dl, Tl, Ur, Dr, Tr, U, D, T, tmp1, tmp2, greens
        # curr_U is used only if the stack is rebuilt
        # this leaves uts.tmp (and curr_U)

        calculate_greens_full1!(it.mc, it.mc.ut_stack, l, 0)
        copyto!(s.curr_U, uts.greens)
        calculate_greens_full2!(it.mc, it.mc.ut_stack, 0, l)
        copyto!(uts.tmp, uts.greens)
        calculate_greens_full1!(it.mc, it.mc.ut_stack, l, l)

        # input and output can be the same here
        copyto!(uts.T, uts.greens)
        Gll = _greens!(it.mc, uts.greens, uts.T, s.tmp2) 
        udt_AVX_pivot!(uts.U, uts.D, uts.T, s.pivot, s.tempv)

        Gl0 = _greens!(it.mc, s.tmp1, s.curr_U, s.tmp2)
        copyto!(s.Tl, s.curr_U)
        udt_AVX_pivot!(s.Ul, s.Dl, s.Tl, s.pivot, s.tempv)
        
        G0l = _greens!(it.mc, s.tmp2, uts.tmp, s.curr_U)
        copyto!(s.Tr, uts.tmp)
        udt_AVX_pivot!(s.Ur, s.Dr, s.Tr, s.pivot, s.tempv)

        return ((G0l, Gl0, Gll), l+1)

    elseif l % it.mc.p.safe_mult == 0
        # Stabilization        
        # Reminder: These overwrite s.tmp1 and s.tmp2
        multiply_slice_matrix_left!(it.mc, it.mc.model, l, s.Ul) 
        multiply_slice_matrix_inv_right!(it.mc, it.mc.model, l, s.Tr)
        multiply_slice_matrix_left!(it.mc, it.mc.model, l, uts.U)
        multiply_slice_matrix_inv_right!(it.mc, it.mc.model, l, uts.T)

        # Gl0
        vmul!(s.tmp1, s.Ul, Diagonal(s.Dl))
        vmul!(s.tmp2, s.tmp1, s.Tl)
        udt_AVX_pivot!(s.Ul, s.Dl, s.tmp1, s.pivot, s.tempv)
        vmul!(s.curr_U, s.tmp1, s.Tl)
        copyto!(s.Tl, s.curr_U)
        Gl0 = _greens!(it.mc, s.tmp1, s.tmp2, s.curr_U)
        # tmp1 must not change anymore
        
        # G0l
        vmul!(s.curr_U, Diagonal(s.Dr), s.Tr)
        vmul!(uts.greens, s.Ur, s.curr_U)
        copyto!(s.Tr, s.curr_U)
        udt_AVX_pivot!(s.tmp2, s.Dr, s.Tr, s.pivot, s.tempv)
        vmul!(s.curr_U, s.Ur, s.tmp2)
        copyto!(s.Ur, s.curr_U)
        G0l = _greens!(it.mc, s.tmp2, uts.greens, s.curr_U)
        # tmp1, tmp2 must not change anymore
        
        # Gll
        vmul!(uts.tmp, uts.U, Diagonal(uts.D))
        vmul!(uts.greens, uts.tmp, uts.T)
        udt_AVX_pivot!(s.curr_U, uts.D, uts.tmp, it.mc.s.pivot, it.mc.s.tempv)
        vmul!(uts.U, uts.tmp, uts.T)
        vmul!(uts.T, Diagonal(uts.D), uts.U)
        udt_AVX_pivot!(uts.tmp, uts.D, uts.T, it.mc.s.pivot, it.mc.s.tempv)
        vmul!(uts.U, s.curr_U, uts.tmp)
        Gll = _greens!(it.mc, uts.greens, uts.greens, s.curr_U)

        return ((G0l, Gl0, Gll), l+1)
    else
        # Quick advance
        multiply_slice_matrix_left!(it.mc, it.mc.model, l, s.Ul) 
        # multiply_slice_matrix_inv_left!(it.mc, it.mc.model, nslices(it.mc)-l+1, s.Ur)
        multiply_slice_matrix_inv_right!(it.mc, it.mc.model, l, s.Tr)
        multiply_slice_matrix_left!(it.mc, it.mc.model, l, uts.U)
        multiply_slice_matrix_inv_right!(it.mc, it.mc.model, l, uts.T)

        # Gl0
        vmul!(s.curr_U, s.Ul, Diagonal(s.Dl))
        vmul!(s.tmp2, s.curr_U, s.Tl)
        Gl0 = _greens!(it.mc, s.tmp1, s.tmp2, s.curr_U)

        # G0l
        vmul!(s.curr_U, s.Ur, Diagonal(s.Dr))
        vmul!(uts.greens, s.curr_U, s.Tr)
        G0l = _greens!(it.mc, s.tmp2, uts.greens, s.curr_U)

        # Gll
        vmul!(s.curr_U, uts.U, Diagonal(uts.D))
        vmul!(uts.greens, s.curr_U, uts.T)
        Gll = _greens!(it.mc, uts.greens, uts.greens, s.curr_U)

        return ((G0l, Gl0, Gll), l+1)
    end
end

function accuracy(iter::CombinedGreensIterator)
    mc = iter.mc
    Gk0s = [deepcopy(greens(mc, k, 0)) for k in 0:nslices(mc)-1]
    [maximum(abs.(Gk0s[i] .- G)) for (i, G) in enumerate(iter)]
end
