using LinearAlgebra
using SparseArrays
using CUDA
using CUDA.CUSPARSE
using Krylov
using Plots
using Printf

include("GPU_2D_funcs.jl")
include("2D_functions.jl")

L = 2.0
k = 1.0
N = 32
u0 = rand(Float64, N, N)
dt = 0.01

println("Warming up JIT...")
smoothing_backward(L, N, k, u0, dt)
smoothing_backward_gpu(L, N, k, u0, dt, stdev=false)
CUDA.synchronize()
println("JIT warm-up done.\n")

sizes     = [32, 64, 128, 256, 512, 1024]

times_serial = Float64[]
times_gpu    = Float64[]

for N in sizes
    u0 = rand(N, N)
    println("Benchmarking N=$N...")

    t_serial = @elapsed smoothing_backward(L, N, k, u0, dt, nt=500)
    push!(times_serial, t_serial)
    println("  serial : $(round(t_serial, digits=4)) s")

    t_gpu = @elapsed smoothing_backward_gpu_no_extra(L, N, k, u0, dt, nt=500)
    push!(times_gpu, t_gpu)
    println("  gpu    : $(round(t_gpu, digits=4)) s")

    println("  speedup: $(round(t_serial/t_gpu, digits=2))×\n")
end

speedups = times_serial ./ times_gpu

p1 = plot(
    sizes, times_serial,
    label="Serial (CPU)", marker=:circle, lw=2, color=:steelblue,
    xlabel="Grid size N (N×N)", ylabel="Wall time (s)",
    title="Backward Euler: Serial vs GPU (Krylov.cg)",
    yscale=:log10, xscale=:log2,
    xticks=(sizes, string.(sizes)),
    legend=:topleft, grid=true, minorgrid=true,
)
plot!(p1, sizes, times_gpu,
    label="GPU (Krylov.cg)", marker=:square, lw=2, color=:crimson)

p2 = bar(
    string.(sizes), speedups,
    xlabel="Grid size N", ylabel="Speedup (serial / GPU)",
    title="GPU Speedup over Serial",
    color=:mediumseagreen, legend=false,
    ylims=(0, max(maximum(speedups)*1.2, 2.0)),
    bar_width=0.5,
)
hline!(p2, [1.0], lw=1.5, ls=:dash, color=:black, label="1× baseline")

final_plot = plot(p1, p2, layout=(2,1), size=(700, 800), dpi=150,
                  left_margin=5Plots.mm, bottom_margin=5Plots.mm)

savefig(final_plot, "benchmark_results.png")
println("Saved benchmark_results.png")

println("\n=== Summary ===")
println("  N   | Serial (s) | GPU (s)  | Speedup")
println("------+------------+----------+--------")
for (i, N) in enumerate(sizes)
    @printf("  %-4d| %-10.4f | %-8.4f | %.2fx\n",
            N, times_serial[i], times_gpu[i], speedups[i])
end