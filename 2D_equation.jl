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
# ------------------------------------------------------------
function smoothing(L, N, k, u0)

    dx = L / (N - 1)
    dy = dx
    h = dx

    dt = 0.24 * h^2 / k # so the forward euler is stable
    nt = 300

    u = copy(u0)
    u_new = similar(u0)
    U = Vector{Matrix{Float64}}(undef, nt)

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

        U[n] = copy(u_new)
        u, u_new = u_new, u
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

# Gaussian initial condition
A = 20.0        # height of the bump
sig = 0.5        # width (smaller = sharper)

u0 = [A * exp(-((x[i] - L / 2)^2 + (y[j] - L / 2)^2) / (2sig^2))
      for i in 1:N, j in 1:N]

U, dt = smoothing(L, N, 1.0, u0)


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
        clims=clims,                     # ← freezes the color scale
        title="t = $(round(n*dt, digits=3))",
    )
end


gif(anim, "heat2d.gif", fps=20)