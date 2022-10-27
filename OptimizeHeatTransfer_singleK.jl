
module OptimizeHeatTransfer_singleK

using Plots
using Printf
using ForwardDiff
using ReverseDiff
using Optim
using Statistics
using UnPack
using Parameters
using TimerOutputs

""" 
Parameter structure
"""
@with_kw struct param
    CFL = 0.2
    Lx
    Ly
    Nx
    Ny
    tfinal
    tol
    Sink_xmin
    Sink_xmax
    T_bot
    T_top
    source
    verbose
    plotFreq = 100
end

""" 
Grid structure
"""
@with_kw struct grid
    x
    y 
    xm
    ym
    dx
    dy
end

"""
Harmonic Mean of two numbers
"""
function hmean(a,b)
    return 2*a*b/(a+b)
end


"""
Solve PDE with own ODE time integrator - Input is C₀ (IC)
"""
function solve_pde(k::Typ, p::param, g::grid) where {Typ}

    @unpack CFL, Nx, Ny, tfinal, Sink_xmin, Sink_xmax, T_bot, T_top, source, verbose = p
    @unpack xm, ym, dx, dy = g

    # Deal with negative k 
    k < eps(0.0) && error("solve_pde does not work with negative or zero k")

    # Preallocate
    dT = zeros(Typ,Nx,Ny)
    fx = zeros(Typ,Nx+1,Ny)
    fy = zeros(Typ,Nx,Ny+1)

    # Initial condition
    t = 0.0
    T = ones(Typ,Nx,Ny)*(T_bot+T_top)*0.5
    
    # Determine timestep
    CFL=0.2
    dt=CFL*dx^2/maximum(k)

    # Number of time iterations
    nStep=ceil(tfinal/dt)

    # Recompute dt with this nStep
    dt=tfinal/nStep

    for iter in 1:nStep

        # Update time
        t = t + dt

        # Compute x-face fluxes = k*dC/dx [interior faces]
        @inbounds for i in 2:Nx, j in 1:Ny
            fx[i,j] = k*(T[i,j] - T[i-1,j]) / dx
        end
        # Compute y-face fluxes = k*dC/dy [interior faces]
        @inbounds for i in 1:Nx, j in 2:Ny
            fy[i,j] = k*(T[i,j] - T[i,j-1]) / dy
        end
        # Heat sinks
        @inbounds for i in 1:Nx
            if xm[i] >= Sink_xmin && xm[i] <= Sink_xmax
                j=   1; fy[i,j] = k*(T[i,j] - T_bot   ) / (0.5*dy)
                j=Ny+1; fy[i,j] = k*(T_top  - T[i,j-1]) / (0.5*dy)
            end
        end
        
        # Compute RHS dC/dt
        @inbounds for i in 1:Nx, j in 1:Ny
            dT[i,j]= ( (fx[i+1,j] - fx[i,j]) / dx 
                     + (fy[i,j+1] - fy[i,j]) / dy 
                     + source )
        end

        # Update C
        T += dt * dT

        # Check if converged 
        maximum(abs.(dT)) > p.tol || return T

        # Outputs
        if verbose
            if rem(iter,p.plotFreq)==0
                @printf("t = %6.3f, max(dT) = %6.3g \n",t,maximum(abs.(dT)))
                myplt = plot(xm,ym,T,st=:surface)
                myplt = plot!(title=@sprintf("Time = %6.3f",t))
                display(myplt)
            end
        end
        
    end

    return T
end

"""
Setup problem to test
"""
function probelmSetup(; Ngrid=50, verbose=false)
    # Inputs
    p=param(
        Lx = 2.0,
        Ly = 2.0,
        Nx = Ngrid,
        Ny = Ngrid,
        tfinal = 10.0,
        tol = 1e-10,
        Sink_xmin = 0.8,
        Sink_xmax = 1.2,
        T_bot = 300,
        T_top = 300,
        source = 1.0,
        verbose = verbose,
    )

    # Grid
    x = range(0.0, p.Lx, length=p.Nx+1)
    y = range(0.0, p.Ly, length=p.Ny+1)
    xm = 0.5 * (x[1:p.Nx] + x[2:p.Nx+1])
    ym = 0.5 * (y[1:p.Ny] + y[2:p.Ny+1])
    dx = x[2] - x[1]
    dy = y[2] - y[1]
    g=grid(x=x,y=y,xm=xm,ym=ym,dx=dx,dy=dy)

    # Initial guess for conductivity k
    k_guess=1.0

    return p,g,k_guess
end


"""
Define cost function to optimize (minimize)
"""
function costFun(k,p,g)
    # Compute C using my own ODE solver
    T=solve_pde(k,p,g)

    # Compute cost (error)
    cost=0.0
    cost += maximum(T)-mean(T) # Want to avoid large temperatures
    cost += maximum(k) # Want to avoid large conductivities
    return cost
end

"""
Optimization - Newton's Method w/ various AD backends
"""
function optimize_Own(f,g!,k,tol; verbose=false)
    verbose && println("\nSolving Optimization Problem with own optimizer")

    # Set AD backend 
    α=1.0 #0.9 # Slow down convergence to compare methods
    iter=0
    converged = false
    G = similar(k)
    while converged == false
        iter += 1

        value = f(k)
        g!(G,k)

        # Update 
        k -= 0.5*G

        # Check if converged
        converged = (abs(value) < tol || iter == 500 || maximum(abs.(G)) < tol)

        # Output for current IC
        verbose && @printf(" %5i, k = %15.5g, Cost Function = %15.6g, max(∇) = %15.6g \n",iter,k[1],value,maximum(abs.(G))) 
    end

    return k # Optimized IC
end

"""
Optimization - Optim.jl
"""
function optimize_Optim(f,g!,k₀,tol; verbose=false)
    verbose && println("\nSolving Optimization Problem with Optim and AD")
    lower = [eps(0.0)]
    upper = [Inf]
    k = Optim.minimizer(optimize(f, g!, lower, upper, k₀, Fminbox(GradientDescent()),
        Optim.Options(
            g_tol = tol,
            iterations = 10000,
            store_trace = false,
            show_trace = verbose,
            #extended_trace = true,
            )))
    return k # Optimized IC
end

"""
Setup function and gradients to use Optimizers
"""
function optimSetup(k,p,g,; ADmethod="Reverse",chunk=50)
    # Function value
    f = k -> costFun(k[1],p,g)

    # Choose method based on ADmethod input
    if ADmethod == "Forward"
        # ForwradDiff Gradient
        tag = ForwardDiff.Tag(f, eltype(k))
        cfg = ForwardDiff.GradientConfig(f, k, ForwardDiff.Chunk{min(chunk,prod(size(k)))}(), tag)
        function g_for!(G,k) 
            G[:] = ForwardDiff.gradient(f,k,cfg)
        end
        g! = (G,k) -> g_for!(G,k)
    elseif ADmethod == "Reverse"
        # ReverseDiff Gradient
        # f_tape = ReverseDiff.GradientTape(f,k)
        # compiled_f_tape = ReverseDiff.compile(f_tape)
        cfg = ReverseDiff.GradientConfig(k)
        function g_rev!(G,k)
            # ReverseDiff.gradient!(G,compiled_f_tape,k)
            ReverseDiff.gradient!(G, f, k, cfg)
        end
        g! = (G,k) -> g_rev!(G,k)
    else
        error("Unknown ADmethod")
    end

    return f,g!
end 

# Test running the PDE solver
function test_methods()
    #p,g,k_guess = probelmSetup(Ngrid=20, verbose=true)
    #T = solve_pde(k_guess,p,g)

    # Setup Optimization problem
    p,g,k_guess = probelmSetup(Ngrid=20, verbose=false)
    f, g! = optimSetup([k_guess,], p, g, ADmethod="Forward")
    
    # Test evaluating f 
    println("f(k_guess) =",@time f(k_guess))

    # Test computing gradient 
    G=[0.0]; @time g!(G,[k_guess,])
    println("grad = ",G)

    # Run Optimizers
    k_Optim = optimize_Optim(f,g!,[k_guess,],p.tol,verbose=true);
    k_Own   = optimize_Own(  f,g!,[k_guess,],p.tol,verbose=true)
    println("Optim")
    println(" -    optimum k = ",k_Optim[1])
    println(" - f(k_optimum) = ",f(k_Optim))
    println("Own")
    println(" -    optimum k = ",k_Own)
    println(" - f(k_optimum) = ",f(k_Own))
end
#test_methods()

# Time function & gradient evaluations
function time_methods()

    grids = [10,20,40,80,100]
    teval = zeros(length(grids))
    tgrad = zeros(length(grids))
    G=[0.0]
    iter = 0
    for grid in grids
        iter += 1
        p,g,k_guess = probelmSetup(Ngrid=grid, verbose=false)
        f, g! = optimSetup([k_guess,], p, g, ADmethod="Forward")

        # Test evaluating f and grad
        teval[iter] = @elapsed f(k_guess)
        println("teval=",teval[1:iter])
        tgrad[iter] = @elapsed g!(G,[k_guess,])
        println("tgrad=",tgrad[1:iter])
    end
end
time_methods()

end # module