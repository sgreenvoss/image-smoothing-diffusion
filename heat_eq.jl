using LinearAlgebra
using cuda

"""
    my_FE_system(T, dt, A, b, y0)
Implements forward Euler on a general ODE system y' = Ay + b(t) with time step
'dt', where 'A' is an m*m matrix, 'b' is a function of t, and 'y0' is an initial 
condition on the time interval 0...'T'.

Returns vector 't' and matrix 'Y' containing the time grid and numerical
approximation, respectively.
"""
function my_FE_system(T, dt, A, b, y0)
    N = Int(round(T / dt))
    t = zeros(N + 1)
    Y = zeros(length(y0), N + 1)
    Y[:, 1] = y0
    for n = 1:N
        t[n+1] = t[n] + dt
        Y[:, n+1] = Y[:, n] + dt * (A*Y[:, n] + b(t[n]))
    end
    
    return t, Y
end

"""
    my_BE_system(T, dt, A, b, y0)
Implements backward Euler on a general ODE system y' = Ay + b(t) with time step
'dt', where 'A' is an m*m matrix, 'b' is a function of t, and 'y0' is an initial 
condition on the time interval 0...'T'.

Returns vector 't' and matrix 'Y' containing the time grid and numerical
approximation, respectively.
"""
function my_BE_system(T, dt, A, b, y0)
    N = Int(round(T / dt))
    t = zeros(N + 1)
    Y = zeros(length(y0), N + 1)
    Y[:, 1] = y0
    C = I - dt * A
    
    for n = 1:N
        t[n+1] = t[n] + dt
        Y[:, n+1] = C \ (Y[:, n] + dt * b(t[n+1]))
    end
    
    return t, Y
end


function step_FE_kernel !( Y , Y_new , dt , dx , N )
    M = length ( Y )
    dx2 = dx ^2
    threads = 256
    blocks = cld (M , threads )

    for n in 0: N -1
        t = n * dt
        @cuda threads = threads blocks = blocks fe_kernel !(
        Y_new , Y , dt , dx2 , K , t )
        Y , Y_new = Y_new , Y
    end
    return Y
end