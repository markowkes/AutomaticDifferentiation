module OptimizeHeatTransfer_singleK

export probelmSetup, optimSetup, optimize_Optim, optimize_Own, plot_taylor

using Plots
using Printf
using ForwardDiff
using ReverseDiff
using Optim
using Statistics
using UnPack
using Parameters
using TimerOutputs
include("Tools/DualTools.jl")
include("Tools/Taylor.jl")

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
    pde_verbose
    makePlot
    outFreq = 100
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
function solve_pde(k::AbstractVector{Typ}, p::param, g::grid) where {Typ}

    @unpack CFL, Nx, Ny, tfinal, Sink_xmin, Sink_xmax, T_bot, T_top, source, pde_verbose, makePlot = p
    @unpack x, y, xm, ym, dx, dy = g

    # Deal with negative k 
    #k[1] < eps(0.0) && error("solve_pde does not work with negative or zero k")

    # Preallocate
    dT = zeros(Typ,Nx,Ny)
    fx = zeros(Typ,Nx+1,Ny)
    fy = zeros(Typ,Nx,Ny+1)

    # Initial condition
    t = 0.0
    T = ones(Typ,Nx,Ny)*(T_bot+T_top)*0.5
    
    # Construct k's on faces of cells using parameratized k 
    kx=zeros(Typ,Nx+1,Ny)
    ky=zeros(Typ,Nx,Ny+1)
    for i in 1:Nx+1, j in 1:Ny 
        kx[i,j] = taylor(k,x[i],ym[j])
    end
    for i in 1:Nx, j in 1:Ny+1
        ky[i,j] = taylor(k,xm[i],y[j])
    end

    # Determine timestep
    CFL=0.2
    dt=CFL*dx^2/max(maximum(kx),maximum(ky))

    # Number of time iterations
    nStep=ceil(tfinal/dt)

    # Recompute dt with this nStep
    dt=tfinal/nStep

    #plot_taylor(k,x,y)

    for iter in 1:nStep

        # Update time
        t = t + dt

        # Compute x-face fluxes = k*dC/dx [interior faces]
        for i in 2:Nx, j in 1:Ny
            fx[i,j] = kx[i,j]*(T[i,j] - T[i-1,j]) / dx
        end
        # Compute y-face fluxes = k*dC/dy [interior faces]
        for i in 1:Nx, j in 2:Ny
            fy[i,j] = ky[i,j]*(T[i,j] - T[i,j-1]) / dy
        end

        # Heat sinks
        for i in 1:Nx
            if xm[i] >= Sink_xmin && xm[i] <= Sink_xmax
                j=   1; fy[i,j] = ky[i,j]*(T[i,j] - T_bot   ) / (0.5*dy)
                j=Ny+1; fy[i,j] = ky[i,j]*(T_top  - T[i,j-1]) / (0.5*dy)
            end
        end

        # Compute RHS dC/dt
        for i in 1:Nx, j in 1:Ny
            dT[i,j]= ( (fx[i+1,j] - fx[i,j]) / dx 
                     + (fy[i,j+1] - fy[i,j]) / dy 
                     + source )
        end

        # Update C
        T += dt * dT

        # Check if converged 
        maximum(abs.(dT)) > p.tol || return T

        # Outputs
        if pde_verbose
            if rem(iter,p.outFreq)==0
                @printf("t = %6.3f, max(dT) = %6.3g \n",t,maximum(abs.(dT)))
                if makePlot
                    # Deal with Duals
                    if typeof(T[1]) <: ForwardDiff.Dual
                        Ts = map(T -> T.value,T)
                    else
                        Ts = T
                    end
                    myplt = plot(xm,ym,Ts',st=:surface)
                    myplt = plot!(title=@sprintf("Time = %6.3f",t))
                    myplt = plot!(xlabel = "x")
                    myplt = plot!(ylabel = "y")
                    myplt = plot!(zlabel = "f(x,y)")
                    display(myplt)
                end
            end
        end
        
    end

    return T
end

"""
Setup problem to test
"""
function probelmSetup(; Ngrid=50, Nk=3, pde_verbose=false, makePlot=false)
    # Inputs
    p=param(
        Lx = 2.0,
        Ly = 2.0,
        Nx = Ngrid,
        Ny = Ngrid,
        tfinal = 10.0,
        tol = 1e-5,
        Sink_xmin = -0.4,
        Sink_xmax =  0.4,
        T_bot = 300,
        T_top = 300,
        source = 1.0,
        pde_verbose = pde_verbose,
        makePlot = makePlot, # Requires pde_verbose to also be true
    )

    # Grid
    x = range(-0.5p.Lx, 0.5p.Lx, length=p.Nx+1)
    y = range(-0.5p.Ly, 0.5p.Ly, length=p.Ny+1)
    xm = 0.5 * (x[1:p.Nx] + x[2:p.Nx+1])
    ym = 0.5 * (y[1:p.Ny] + y[2:p.Ny+1])
    dx = x[2] - x[1]
    dy = y[2] - y[1]
    g=grid(x=x,y=y,xm=xm,ym=ym,dx=dx,dy=dy)

    # Initial guess for conductivity k
    k_guess=zeros(Nk)
    k_guess[1]=0.1

    return p,g,k_guess
end


"""
Define cost function to optimize (minimize)
"""
function costFun(k::AbstractVector{Typ},p,g) where {Typ}

    # Compute C using my own ODE solver
    T=solve_pde(k,p,g)

    # Compute cost (error)
    cost=0.0
    cost += sum(exp.(T.-mean(T))) # Want to avoid large temperatures

    # Evalute ks on grid 
    ks=zeros(Typ,p.Nx,p.Ny)
    for i in 1:p.Nx, j in 1:p.Ny 
        ks[i,j] = taylor(k,g.xm[i],g.ym[j])
    end
    cost += sum(exp.(ks))            # Avoid large conductivities
    #cost += sum(max.(-log.(ks),0.0)) # Avoid very small conductivities
    cost += sum(exp.(-1e3ks))
    return cost
end

"""
Optimization - Newton's Method w/ various AD backends
"""
function optimize_Own(fg!,k,tol; optim_verbose=false)
    optim_verbose && println("\nSolving Optimization Problem with own optimizer")

    # Set AD backend 
    α=1e-3 #0.9 # Slow down convergence to compare methods
    iter=0
    converged = false
    F = similar(k)
    G = similar(k)
    while converged == false
        iter += 1

        F = fg!(F,G,k)

        # Update 
        k -= α*G

        # Limit k[1]
        k[1] = max(eps(0.0),k[1])

        # Check if converged
        converged = (abs(F) < tol || iter == 500 || maximum(abs.(G)) < tol)

        # Output for current IC
        optim_verbose && @printf(" %5i, k = %15.5g, Cost Function = %15.6g, max(∇) = %15.6g \n",iter,k[1],F,maximum(abs.(G))) 
    end

    return k # Optimized IC
end

"""
Optimization - Optim.jl
"""
function optimize_Optim(fg!,k₀,tol; optim_verbose=false)
    optim_verbose && println("\nSolving Optimization Problem with Optim and AD")
    # lower =-Inf*ones(size(k₀))
    # lower[1] = eps(0.0)
    # upper = Inf*ones(size(k₀))
    # k = Optim.minimizer(optimize(Optim.only_fg!(fg!), lower, upper, k₀, Fminbox(GradientDescent()),
    #     Optim.Options(
    #         g_tol = tol,
    #         iterations = 100,
    #         store_trace = false,
    #         show_trace = optim_verbose,
    #         #extended_trace = true,
    #         )))
    k = Optim.minimizer(optimize(Optim.only_fg!(fg!), k₀, GradientDescent(),
            Optim.Options(
                g_tol = tol,
                iterations = 100,
                store_trace = false,
                show_trace = optim_verbose,
                #extended_trace = true,
                )))
    return k # Optimized IC
end

"""
Setup function and gradients to use Optimizers
"""
function optimSetup(k,p,g,; ADmethod="Forward",chunk=50)
    # Function value
    f = k -> costFun(k,p,g)

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
    #p,g,k_guess = probelmSetup(Ngrid=20, pde_verbose=true)
    #T = solve_pde(k_guess,p,g)

    # Setup Optimization problem
    p,g,k_guess = probelmSetup(Ngrid=10, Nk=Taylor_nBasis_order(2), pde_verbose=false, makePlot=false)
    f,fg! = optimSetup(k_guess, p, g, ADmethod="Forward")

    # Test computing value 
    #@time value = f(k_guess)
    #println("value = ",value)

    # Test computing value and gradient
    # println("Calling fg! with k=",k_guess)
    # F=0.0; G=zeros(size(k_guess)); @time F = fg!(F,G,k_guess)
    # println("value = ",F)
    # println(" grad = ",G)

    # Run Optimizers
    k_Optim = optimize_Optim(fg!,k_guess,p.tol,optim_verbose=true);
    k_Own   = optimize_Own(  fg!,k_guess,p.tol,optim_verbose=true)

    println("Optim")
    println(" -    optimum k = ",k_Optim)
    println(" - f(k_optimum) = ",f(k_Optim))
    plot_taylor(k_Optim,g.xm,g.ym)
    
    println("Own")
    println(" -    optimum k = ",k_Own)
    println(" - f(k_optimum) = ",f(k_Own))
    plot_taylor(k_Own,g.xm,g.ym)

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
        p,g,k_guess = probelmSetup(Ngrid=grid, pde_verbose=false)
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