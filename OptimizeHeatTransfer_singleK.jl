
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
function optimize_Own(fg!,k,tol; verbose=false)
    verbose && println("\nSolving Optimization Problem with own optimizer")

    # Set AD backend 
    α=1.0 #0.9 # Slow down convergence to compare methods
    iter=0
    converged = false
    F = similar(k)
    G = similar(k)
    while converged == false
        iter += 1

        # Compute function falue and gradient
        F = fg!(F,G,k)

        # Update 
        k -= 0.5*G

        # Check if converged
        converged = (abs(F) < tol || iter == 500 || maximum(abs.(G)) < tol)

        # Output for current IC
        verbose && @printf(" %5i, k = %15.5g, Cost Function = %15.6g, max(∇) = %15.6g \n",iter,k[1],F,maximum(abs.(G))) 
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

function optimize_Optim(fg!,k₀,tol; verbose=false)
    verbose && println("\nSolving Optimization Problem with Optim and AD")
    lower = [eps(0.0)]
    upper = [Inf]
    k = Optim.minimizer(optimize(Optim.only_fg!(fg!), lower, upper, k₀, Fminbox(LBFGS()),
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
        # Value and Gradient
        results = DiffResults.GradientResult(k)
        tag = ForwardDiff.Tag(f, eltype(k))
        cfg = ForwardDiff.GradientConfig(f, k, ForwardDiff.Chunk{min(chunk,prod(size(k)))}(), tag)
        function fg_for!(F,G,k)
            ForwardDiff.gradient!(results,f,k,cfg)
            if F !== nothing
                F = DiffResults.value(results)
            end
            if G !== nothing 
                G[:] = DiffResults.gradient(results)
            end
            return F
        end
        fg! = (F,G,k) -> fg_for!(F,G,k)

    elseif ADmethod == "Reverse"
        # ReverseDiff Gradient
        results = DiffResults.GradientResult(k)
        cfg = ReverseDiff.GradientConfig(k)
        function fg_rev!(F,G,k)
            ReverseDiff.gradient!(results, f, k, cfg)
            if F !== nothing
                F = DiffResults.value(results)
            end
            if G !== nothing 
                G[:] = DiffResults.gradient(results)
            end
            return F
        end
        fg! = (F,G,k) -> fg_rev!(F,G,k)
    else
        error("Unknown ADmethod")
    end

    return f,fg!
end 

# Test running the PDE solver
function test_methods()
    #p,g,k_guess = probelmSetup(Ngrid=20, verbose=true)
    #T = solve_pde(k_guess,p,g)

    # Setup Optimization problem
    p,g,k_guess = probelmSetup(Ngrid=10, verbose=false)
    f,fg! = optimSetup([k_guess,], p, g, ADmethod="Forward")
    
    # Test computing value and gradient 
    F=0.0; G=[0.0,]; @time fg!(F,G,[k_guess,])
    println("value = ",F,"grad = ",G)

    # Run Optimizers
    k_Optim = optimize_Optim(fg!,[k_guess,],p.tol,verbose=true);
    k_Own   = optimize_Own(  fg!,[k_guess,],p.tol,verbose=true)
    println("Optim")
    println(" -    optimum k = ",k_Optim[1])
    println(" - f(k_optimum) = ",f(k_Optim[1]))
    println("Own")
    println(" -    optimum k = ",k_Own[1])
    println(" - f(k_optimum) = ",f(k_Own))
end
test_methods()

# Time function & gradient evaluations
function time_methods()

    # Grids to test
    grids = [10,20,40,80,100]

    # Preallocate time arrays
    teval = zeros(length(grids))
    tgrad = zeros(length(grids))
    tboth = zeros(length(grids))

    # Initialize value and gradient arrays
    F=[0.0,]
    G=[0.0,]
    
    # Iterate over grids
    iter = 0
    for grid in grids
        iter += 1

        # Setup problem for this grid
        p,g,k_guess = probelmSetup(Ngrid=grid, verbose=false)
        f, g!, fg! = optimSetup([k_guess,], p, g, ADmethod="Forward")

        # Test evaluating f
        teval[iter] = @elapsed f(k_guess)
        println("teval=",teval[1:iter])

        # Test evaluating g!
        tgrad[iter] = @elapsed g!(G,[k_guess,])
        println("tgrad=",tgrad[1:iter])

        # Test evaluating fg!
        tboth[iter] = @elapsed fg!(F,G,[k_guess,])
        println("tgrad=",tboth[1:iter])
    end
end
#time_methods()

end # module