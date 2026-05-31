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

function smoothing(L, N, k, u0, T=0.1)
    save_per = 100

    dx = L / (N - 1)
    dy = dx
    h = dx

    dt = 0.24 * h^2 / k # so the forward euler is stable
    nt = Int(round(T/dt))

    u = copy(u0)
    u_new = similar(u0)
    U = Vector{Matrix{Float64}}()

    for n in 1:nt
        apply_neumann!(u)
        #apply_dirchlet!(u)

        @inbounds for i in 2:N-1, j in 2:N-1
            uxx = (u[i+1, j] - 2u[i, j] + u[i-1, j]) / dx^2
            uyy = (u[i, j+1] - 2u[i, j] + u[i, j-1]) / dy^2
            u_new[i, j] = u[i, j] + dt * k * (uxx + uyy)
        end

        apply_neumann!(u_new)
        #apply_dirchlet!(u_new)

        if n % save_per == 0
            push!(U, copy(u_new))
        end

        u, u_new = u_new, u
    end

    return U, dt * save_per
end