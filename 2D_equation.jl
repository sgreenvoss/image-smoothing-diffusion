using LinearAlgebra
using Plots
using SparseArrays

# ------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------
function apply_neumann!(u)
    Nx, Ny = size(u)

    u[1, :] .= u[2, :]      # left
    u[Nx, :] .= u[Nx-1, :]    # right

    u[:, 1] .= u[:, 2]       # bottom
    u[:, Ny] .= u[:, Ny-1]    # top

    return nothing
end

function apply_dirichlet!(u)
    u[1, :] .= 0.0
    u[end, :] .= 0.0
    u[:, 1] .= 0.0
    u[:, end] .= 0.0
    return nothing
end

# ------------------------------------------------------------
# Build 2D Laplacian with BCs
# ------------------------------------------------------------
"""
    laplacian_neumann(N, h)

Constructs the 2D Laplacian with Neumann boundary conditions.

# Arguments
- `N::Int`    — number of grid points in each spatial direction
- `h::Float`  — grid spacing

# Returns
- `L::SparseMatrixCSC` — discrete Laplacian with Neumann BCs
"""
function laplacian_neumann(N, h)
    NN = N * N
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    idx(i, j) = (j - 1) * N + i

    for j in 1:N
        for i in 1:N
            k = idx(i, j)
            center = -4.0

            # left
            if i > 1
                push!(rows, k)
                push!(cols, idx(i - 1, j))
                push!(vals, 1.0)
            else
                center += 1.0
            end

            # right
            if i < N
                push!(rows, k)
                push!(cols, idx(i + 1, j))
                push!(vals, 1.0)
            else
                center += 1.0
            end

            # down
            if j > 1
                push!(rows, k)
                push!(cols, idx(i, j - 1))
                push!(vals, 1.0)
            else
                center += 1.0
            end

            # up
            if j < N
                push!(rows, k)
                push!(cols, idx(i, j + 1))
                push!(vals, 1.0)
            else
                center += 1.0
            end

            push!(rows, k)
            push!(cols, k)
            push!(vals, center)
        end
    end

    return sparse(rows, cols, vals) / h^2
end

"""
    laplacian_dirichlet(N, h)

Constructs the 2D Laplacian with Dirichlet
boundary conditions. Boundary rows of the matrix are
replaced with identity rows to enforce the Dirichlet constraint.

# Arguments
- `N::Int`    — number of grid points in each spatial direction
- `h::Float`  — grid spacing

# Returns
- `L::SparseMatrixCSC` — discrete Laplacian with Dirichlet BCs
"""

function laplacian_dirichlet(N, h)
    NN = N * N
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    idx(i, j) = (j - 1) * N + i

    for j in 1:N
        for i in 1:N
            k = idx(i, j)

            if i == 1 || i == N || j == 1 || j == N
                # Boundary → identity row
                push!(rows, k)
                push!(cols, k)
                push!(vals, 1.0)
                continue
            end

            # Interior 5‑point stencil
            push!(rows, k)
            push!(cols, idx(i, j))
            push!(vals, -4.0)
            push!(rows, k)
            push!(cols, idx(i - 1, j))
            push!(vals, 1.0)
            push!(rows, k)
            push!(cols, idx(i + 1, j))
            push!(vals, 1.0)
            push!(rows, k)
            push!(cols, idx(i, j - 1))
            push!(vals, 1.0)
            push!(rows, k)
            push!(cols, idx(i, j + 1))
            push!(vals, 1.0)
        end
    end

    return sparse(rows, cols, vals) / h^2
end


# ------------------------------------------------------------
# Forward Euler smoothing (explicit)
# ------------------------------------------------------------
"""
    smoothing(L, N, k, u0)

Solves the 2D heat equation `u_t = k (u_xx + u_yy)` using the explicit Forward
Euler method on a square domain with Neumann boundary conditions.
The time step `dt` is chosen to satisfy the stability condition for
the explicit scheme.

# Arguments
- `L::Float`     — physical domain length
- `N::Int`       — number of grid points per dimension
- `k::Float`     — diffusion coefficient
- `u0::Matrix`   — initial condition (N by N)

# Returns
- `U::Vector{Matrix}` — solution snapshots over time
- `dt::Float`         — time step used
"""

function smoothing(L, N, k, u0)
    dx = L / (N - 1)
    h = dx

    dt = 0.24 * h^2 / k   # stability for forward Euler
    nt = 300

    u = copy(u0)
    u_new = similar(u0)
    U = Vector{Matrix{Float64}}(undef, nt)

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

        U[n] = copy(u_new)
        u, u_new = u_new, u
    end

    return U, dt
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

# ------------------------------------------------------------
# Problem setup
# ------------------------------------------------------------
L = 2.0
N = 51

x = range(0, L, length=N)
y = range(0, L, length=N)

A0 = 50.0   # height of the bump
sig = 2    # width

u0 = [A0 * exp(-((x[i] - L / 2)^2 + (y[j] - L / 2)^2) / (2sig^2))
      for i in 1:N, j in 1:N]

U, dt = smoothing_backward(L, N, 1.0, u0, 0.01, nt=300)

# ------------------------------------------------------------
# Static plot of final state
# ------------------------------------------------------------
heatmap(
    x, y, U[end]',
    aspect_ratio=1,
    xlabel="x",
    ylabel="y",
    title="Temperature Field u(x,y)",
    c=:viridis,
    colorbar=true,
)

# ------------------------------------------------------------
# Animation
# ------------------------------------------------------------
clims = extrema(U[1])

anim = @animate for n in 1:length(U)
    heatmap(
        x, y, U[n]',
        aspect_ratio=1,
        c=:viridis,
        clims=clims,
        title="t = $(round(n*dt, digits=3))",
    )
end

gif(anim, "heat2d.gif", fps=20)
