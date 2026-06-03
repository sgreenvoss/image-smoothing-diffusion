using LinearAlgebra
using Plots

# ------------------------------------------------------------
# Apply Neumann BCs (zero normal derivative) via ghost-point reflection
# ------------------------------------------------------------
function apply_neumann!(u)
    Nx, Ny = size(u)

    @inbounds for j in 1:Ny
        u[1, j] = u[2, j]        # left
        u[Nx, j] = u[Nx-1, j]     # right
    end

    @inbounds for i in 1:Nx
        u[i, 1] = u[i, 2]        # bottom
        u[i, Ny] = u[i, Ny-1]     # top
    end

    return nothing
end

# ------------------------------------------------------------
# Apply Dirichlet BCs (u = 0 on boundary) 
# ------------------------------------------------------------
function apply_dirchlet!(u)
    u[1, :] .= 0.0
    u[end, :] .= 0.0
    u[:, 1] .= 0.0
    u[:, end] .= 0.0
    return u
end

# ------------------------------------------------------------
# Solve u_t = k (u_xx + u_yy) on a square domain with Neumann BCs
# Returns: U (vector of solution snapshots), dt
# N i think is the size of the domain NxN?
# ------------------------------------------------------------

function smoothing(L, N, k, u0; thresh=0.01)
    dx = L / (N - 1)
    h = dx

    dt = 0.24 * h^2 / k   # stability for forward Euler
    nt = 300

    u = copy(u0)
    u_new = similar(u0)

    for n in 1:nt
        apply_neumann!(u)
        # apply_dirichlet!(u)

        @inbounds for i in 2:N-1, j in 2:N-1
            uxx = (u[i+1, j] - 2u[i, j] + u[i-1, j]) / dx^2
            uyy = (u[i, j+1] - 2u[i, j] + u[i, j-1]) / dx^2
            u_new[i, j] = u[i, j] + dt * k * (uxx + uyy)
        end

        apply_neumann!(u_new)
        # apply_dirichlet!(u_new)

        u, u_new = u_new, u
        if std(u) < thresh
            return 
        end
    end
end


# ------------------------------------------------------------
# Backward Euler smoothing (implicit, with LU factorization)
# ------------------------------------------------------------
"""
    smoothing_backward(L, N, k, u0, dt; nt=300)

Solves the 2D heat equation `u_t = k ∇²u` using the implicit Backward
Euler method. The linear system

    (I - dt*k*L) u^{n+1} = u^n

is solved efficiently using a precomputed LU factorization of the
system matrix. This method is unconditionally stable and suitable for
larger time steps.

# Arguments
- `L::Float`     — physical domain length
- `N::Int`       — number of grid points per dimension
- `k::Float`     — diffusion coefficient
- `u0::Matrix`   — initial condition (N by N)
- `dt::Float`    — time step
- `nt::Int`      — number of time steps (default: 300)

# Returns
- `U::Vector{Matrix}` — solution snapshots over time
- `dt::Float`         — time step (returned unchanged)
"""

function smoothing_backward(L, N, k, u0, dt; nt=300)
    dx = L / (N - 1)

    # Build Laplacian and system matrix
    Lap = laplacian_neumann(N, dx)

    Ibig = sparse(I, N^2, N^2)
    A = Ibig - dt * k * Lap

    F = lu(A)

    # Work with vectorized state
    u_vec = vec(copy(u0))
    U = Vector{Matrix{Float64}}(undef, nt)

    for n in 1:nt
        u_new_vec = F \ u_vec
        u_mat = reshape(u_new_vec, N, N)

        U[n] = copy(u_mat)
        u_vec = vec(u_mat)
    end

    return U, dt
end