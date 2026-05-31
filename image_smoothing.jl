include("2D_functions.jl")
include("image_load.jl")

println("loaded and running.")

function run_with_rand_img()
    x,y = 500, 500
    L = 5.0
    img = rand_img(x, y)

    U, dt = smoothing(L, x, 1.0, img)
    println("finished smoothing")
    #display(U)

    clims = extrema(U[1])
    xr = range(0, L, length=x)
    yr = range(0, L, length=x)


    anim = @animate for n in 1:length(U)
        heatmap(
            xr, yr, U[n]',
            aspect_ratio=1,
            c=:viridis,
            clims=clims,                     # ← freezes the color scale
            title="t = $(round(n*dt, digits=3))",
        )
    end
    gif(anim, "gifs/heat2d-500.gif", fps=20)
end

function run_with_real_img() 
    img = testimage("mandril_gray")
    img = rotr90(img)
    x, y = 512, 512
    L = Float64(x)

    U, dt = smoothing(L, x, 1.0, img, 1000)
    println("finished smoothing")

    clims = extrema(U[1])

    xr = range(0, L, length=x)
    yr = range(0, L, length=x)

    anim = @animate for n in 1:length(U)
        heatmap(
            xr, yr, U[n]',
            aspect_ratio=1,
            c=:viridis,
            clims=clims,                     # ← freezes the color scale
            title="t = $(round(n*dt, digits=3))",
            axis=false,
            colorbar=false,
        )
    end
    gif(anim, "gifs/heat2d-faster.gif", fps=60)
end

run_with_real_img()