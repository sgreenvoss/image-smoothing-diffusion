using LinearAlgebra
using Plots
using SparseArrays
using CUDA
using CUDA.CUSOLVER
using CUDA.CUSPARSE
using Krylov
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

Constructs the 2D Laplacian with Neumann boundary conditions on GPU.

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
# Backward Euler smoothing — GPU LU solve via CUSOLVER
# ------------------------------------------------------------
"""
    smoothing_backward(L, N, k, u0, dt; nt=300, target=0.009, gif_path=nothing, gif_stride=10)
 
Solves the 2D heat equation `u_t = k ∇²u` using implicit Backward Euler:
 
    (I - dt*k*L) u^{n+1} = u^n
 
The system matrix is kept on GPU and solved each step via CUSOLVER's sparse LU.
This avoids the CPU↔GPU round-trip that the serial version pays on every step.
 
Optionally saves a GIF of the solution evolving over time.
 
# Arguments
- `L::Float`        — physical domain length
- `N::Int`          — number of grid points per dimension
- `k::Float`        — diffusion coefficient
- `u0::Matrix`      — initial condition (NxN)
- `dt::Float`       — time step
- `nt::Int`         — number of time steps (default: 300)
- `target::Float`   — std threshold for smoothness detection (default: 0.009)
- `stdev::Bool`     — whether it should create an array of standard deviations (default: true)
- `gif_path`        — if a String path is given, saves a GIF there; nothing skips it
- `gif_stride::Int` — save every nth frame to the GIF (default: 10)
 
# Returns
- `stdevs::Vector{Float64}` — per-step standard deviation of the solution
- `dt::Float`               — time step (returned unchanged)
"""
function smoothing_backward_gpu(L, N, k, u0, dt, nt=300, target=0.009;
                             stdev=true, gif_path=nothing, gif_stride=10)
    dx = L / (N - 1)
    stdevs = Float64[]
 
    Lap_gpu = laplacian_neumann(N, dx)          
    Lap_cpu = SparseMatrixCSC(Lap_gpu)         
    NN = N * N
    Ibig = sparse(I, NN, NN)
    A_cpu = Ibig - dt * k * Lap_cpu
    A_gpu = CuSparseMatrixCSR(A_cpu)          
 
    u_gpu = CuArray(Float64.(vec(copy(u0))))
 
    do_gif = gif_path !== nothing
    local anim
    if do_gif
        anim = Animation()
        clims = extrema(Float64.(u0))
        xr = range(0, L, length=N)
        yr = range(0, L, length=N)
    end
 
    for n in 1:nt
        u_gpu, _ = Krylov.cg(A_gpu, u_gpu)
        
        if stdev
            push!(stdevs, std(Array(u_gpu)))
        end

        if do_gif && n % gif_stride == 0
            u_mat = Array(u_gpu)  
            heatmap(xr, yr, u_mat',
                aspect_ratio=1,
                c=:viridis,
                clims=clims,
                title="t = $(round(n*dt, digits=3))",
                axis=false,
                colorbar=false,
                fps=10)
            frame(anim)
        end
    end
 
    if do_gif
        gif(anim, gif_path, fps=20)
        println("Saved GIF → $gif_path")
    end
 
    return stdevs, dt
end

function smoothing_backward_gpu_no_extra(L, N, k, u0, dt; nt=300)
    dx = L / (N - 1)
    stdevs = Float64[]
 
    # --- Build system matrix entirely on GPU ---
    Lap_gpu = laplacian_neumann(N, dx)          
    Lap_cpu = SparseMatrixCSC(Lap_gpu)         
    NN = N * N
    Ibig = sparse(I, NN, NN)
    A_cpu = Ibig - dt * k * Lap_cpu
    A_gpu = CuSparseMatrixCSR(A_cpu)          
 
    # Initial state on GPU
    u_gpu = CuArray(Float64.(vec(copy(u0))))
 
    for n in 1:nt
        u_gpu, _ = Krylov.cg(A_gpu, u_gpu)
    end

    return dt
end

function run_test()
    images = ["resolution_test_512", "bark_he_512", "cameraman", "mandril_gray"]

    dt = 0.001
    nt = 500
    final_t = dt * nt
    x = range(0, final_t, length=nt)
    plt = plot()
    
    mkpath("gifs")
    
    for image in images
        println("Processing: $image")
        img = testimage(image)
        img = rotr90(img)
        N = 512
        L = 2.0
    
        _, _ = smoothing_backward_gpu(L, N, 1.0, img, dt, nt, 0.009;
                                        gif_path="new_gifs/$image.gif",
                                        gif_stride=2, stdev=false)
    end
end

run_test()