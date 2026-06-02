using LinearAlgebra
using Plots
using SparseArrays
using CUDA
using TestImages
using Statistics

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

function is_smooth(A, target)
    return std(A) < target
end

# ------------------------------------------------------------

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
    max_nnz = 5 * NN
    rows = CUDA.zeros(Int32, max_nnz)
    cols = CUDA.zeros(Int32, max_nnz)
    vals = CUDA.zeros(Float64, max_nnz)
    counts = CUDA.zeros(Int32, NN)

    function kernel(rows, cols, vals, counts, N)
        idx_lin = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if idx_lin > N*N
            return nothing
        end
        j = (idx_lin - 1) ÷ N + 1
        i = (idx_lin - 1) % N + 1
        k = (j - 1) * N + i

        base = (k - 1) * 5
        count = Int32(0)
        center = -4.0

        if i > 1
            count += 1
            rows[base+count] = k 
            cols[base+count] = (j-1) * N + (i-1)
            vals[base+count] = 1.0
        else
            center += 1
        end 

        if i < N 
            count += 1
            rows[base+count] = k 
            cols[base+count] = (j-1) *N + (i+1)
            vals[base+count] = 1.0
        else 
            center += 1.0
        end 

        if j > 1 
            count += 1
            rows[base+count] = k 
            cols[base+count] = (j-2) * N + i 
            vals[base+count] = 1.0
        else 
            center += 1.0
        end 

        if j < N 
            count += 1
            rows[base+count] = k 
            cols[base+count] = j * N + i 
            vals[base+count] = 1.0 
        else 
            center += 1.0 
        end 

        count += 1
        rows[base+count] = k 
        cols[base+count] = k 
        vals[base+count] = center

        counts[k] = count 
        return nothing 
    end

    threads = 256
    blocks = cld(NN, threads)
    @cuda threads=threads blocks=blocks kernel(rows,cols,vals,counts,N)
    CUDA.synchronize()

    counts_cpu = Array(counts)
    rows_cpu   = Array(rows)
    cols_cpu   = Array(cols)
    vals_cpu   = Array(vals)

    total_nnz = sum(counts_cpu)
    final_rows = Vector{Int32}(undef, total_nnz)
    final_cols = Vector{Int32}(undef, total_nnz)
    final_vals = Vector{Float64}(undef, total_nnz)

    ptr = 1
    for k in 1:NN
        base = (k - 1) * 5
        n = counts_cpu[k]
        for s in 1:n
            final_rows[ptr] = rows_cpu[base + s]
            final_cols[ptr] = cols_cpu[base + s]
            final_vals[ptr] = vals_cpu[base + s]
            ptr += 1
        end
    end
    final_vals = final_vals ./ h^2
    l_cpu = sparse(final_rows, final_cols, final_vals, NN, NN)
    return l_cpu
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

function smoothing_backward(L, N, k, u0, dt, nt=300, target=0.009)
    checking = true
    dx = L / (N - 1)
    stdevs = Float64[]

    # Build Laplacian and system matrix
    Lap = laplacian_neumann(N, dx)

    Ibig = sparse(I, N^2, N^2)
    A = Ibig - dt * k * Lap

    F = lu(A)

    # Work with vectorized state
    u_vec = Float64.(vec(copy(u0)))
    U = Vector{Matrix{Float64}}(undef, nt)

    for n in 1:nt
        u_new_vec = F \ u_vec
        u_mat = reshape(u_new_vec, N, N)
        # remove this when done with plotting stdevs
        push!(stdevs, std(u_mat))
        if checking && is_smooth(u_mat, target)
            println("Reached smooth state at step ", n, "at time ", n*dt)
            checking = false
        end

        U[n] = copy(u_mat)
        u_vec = vec(u_mat)
    end

    # remove stdevs when one with plotting stdev
    return U, dt, stdevs 
end

function stdevs_from_img(image_name, size; d_t=0.001, target=0.001, n_t=100)
    img = testimage(image_name)
    img = rotr90(img)
    N = size 
    L = 2.0 

    U, dt, stdevs = smoothing_backward(L, N, 1.0, img, d_t, n_t, target)

    return stdevs
end
# ------------------------------------------------------------
# Problem setup
# ------------------------------------------------------------
images = ["resolution_test_512", "bark_he_512", "cameraman", "mandril_gray"]

dt = 0.001
nt = 500
final_t = dt * nt 
x = range(0, final_t, length=nt)
plt = plot()

for image in images
    s = stdevs_from_img(image, 512, n_t=nt)
    plot!(x, s, label=image)
end
hline!([0.009], label="Smoothing threshold", ls=:dot)
xlabel!("Time (s)")
ylabel!("Standard deviation")
savefig("all_devs_with_thresh.png")

# img = testimage("cameraman")
# img = rotr90(img)
# N = 512
# L = 2.0

# nt = 100

# U, dt, stdevs = smoothing_backward(L, N, 1.0, img, 0.001, 500)

# clims = extrema(U[1])

# xr = range(0, L, length=N)
# yr = range(0, L, length=N)


# anim = @animate for n in 1:length(U)
#     heatmap(
#         xr, yr, U[n]',
#         aspect_ratio=1,
#         c=:viridis,
#         clims=clims,                     # ← freezes the color scale
#         title="t = $(round(n*dt, digits=3))",
#         axis=false,
#         colorbar=false,
#     )
# end

# gif(anim, "gifs/cameraman.gif", fps=20)
