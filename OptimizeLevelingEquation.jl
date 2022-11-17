module OptimizeLevelingEquation

export probelmSetup, optimSetup, optimize_Optim, optimize_Own, plot_taylor, param, grid

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
    μ
    σ
    pde_verbose
    makePlot
    outFreq
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
Solve PDE with own ODE time integrator - Input is C₀ (IC)
"""
function solve_pde(ICparams::AbstractVector{Typ}, p::param, g::grid) where {Typ}

    @unpack CFL, Nx, Ny, tfinal, μ, σ, pde_verbose, makePlot = p
    @unpack x, y, xm, ym, dx, dy = g

    λ  = ICparams[1]
    aₒ = ICparams[2]
    hₒ = ICparams[3]

    # Initial condition
    t=0.0
    h=zeros(Typ,Nx,Ny)
    for i=1:Nx, j=1:Ny
        h[i,j] = hₒ + aₒ * sin(2π*xm[i]/λ + π/2)
    end
    hmin = DualValue(minimum(h))
    hmax = DualValue(maximum(h))

    # Determine timestep
    CFL=0.2
    #dt=CFL*dx^2/max(maximum(kx),maximum(ky))
    dt=1e-3 #CFL*dx^2/1e-2

    # Number of time iterations
    nStep=ceil(tfinal/dt)

    # Recompute dt with this nStep
    dt=tfinal/nStep

    # Periodic boundary conditions helpers
    function per(i,j)
        iper = i
        if i < 1 
            iper = i+Nx
        elseif i > Nx
            iper = i-Nx 
        end
        jper = j
        if Ny==1
            jper = 1
        else
            if j < 1 
                jper = j+Ny
            elseif j > Ny
                jper = j-Ny
            end
        end
        return CartesianIndex(iper,jper)
    end

    # Preallocate work arrays 
    rhs = zeros(Typ,Nx,Ny)

    for iter in 1:nStep

        # Update time
        t += dt

        # Update h
        ∇²xh(i,j) = (h[per(i+1,j)] - 2h[per(i,j)] + h[per(i-1,j)]) / dx^2
        ∇²yh(i,j) = (h[per(i,j+1)] - 2h[per(i,j)] + h[per(i,j-1)]) / dy^2
        ∇²h(i,j)     = ∇²xh(i,j) + ∇²yh(i,j)
        C∇x∇²h(i,j)   = h[per(i,j)]^3.0/3μ*σ*(∇²h(i+1,j) - ∇²h(i-1,j)) / 2dx
        C∇y∇²h(i,j)   = h[per(i,j)]^3.0/3μ*σ*(∇²h(i,j+1) - ∇²h(i,j-1)) / 2dy
        for i=1:Nx, j=1:Ny
            rhs[i,j] = - (
                (C∇x∇²h(i+1,j) - C∇x∇²h(i-1,j)) / 2dx +
                (C∇y∇²h(i,j+1) - C∇y∇²h(i,j-1)) / 2dy )
        end

        h += dt*rhs

        # Outputs
        if pde_verbose
            if rem(iter,p.outFreq)==0
                @printf("t = %6.3g max(rhs) = %6.3g\n",t,DualValue(maximum(abs.(rhs))))
                if makePlot
                    # Deal with Duals
                    if eltype(h) <: ForwardDiff.Dual || eltype(h) <: ReverseDiff.TrackedReal
                        hs = map(h -> h.value,h)
                    else
                        hs = h
                    end
                    if Ny>1
                        myplt = plot(xm,ym,hs',st=:surface)
                        myplt = plot!(title=@sprintf("Time = %6.3f",t))
                        myplt = plot!(xlabel = "x")
                        myplt = plot!(ylabel = "y")
                        myplt = plot!(zlabel = "h(x,y)")
                        myplt = plot!(zlim=(hmin,hmax))
                        display(myplt)
                    else
                        myplot = plot(xm,hs[:,1])
                        myplt = plot!(title=@sprintf("Time = %6.3f",t))
                        myplt = plot!(xlabel = "x")
                        myplt = plot!(ylabel = "h(x,y)")
                        myplt = plot!(ylim=(hmin,hmax))
                        display(myplt)
                    end 
                end
            end
        end
        
    end

    return h
end

"""
Setup problem to test
"""
function probelmSetup(; Ngrid=50, pde_verbose=false, makePlot=false)
    # Inputs
    p=param(
        Lx = 0.004,
        Ly = 0.004,
        Nx = Ngrid,
        Ny = Ngrid,
        tfinal = 10.0,
        tol = 1e-5,
        μ = 0.1, # Pa-s
        σ = 0.03, # N/m
        pde_verbose = pde_verbose,
        makePlot = makePlot, # Requires pde_verbose to also be true
        outFreq = 1000,
    )

    # Grid
    x = range(0.0, p.Lx, length=p.Nx+1)
    y = range(0.0, p.Ly, length=p.Ny+1)
    xm = 0.5 * (x[1:p.Nx] + x[2:p.Nx+1])
    ym = 0.5 * (y[1:p.Ny] + y[2:p.Ny+1])
    dx = x[2] - x[1]
    dy = y[2] - y[1]
    g=grid(x=x,y=y,xm=xm,ym=ym,dx=dx,dy=dy)

    # Initial guess for IC parameters
    hₒ = 125e-6 # m 
    aₒ = 50e-6 # m
    λ = p.Lx
    ICparams = [λ,aₒ,hₒ]

    return p,g,ICparams
end
# Test running solver
#p,g,ICparams = probelmSetup(Ngrid=10,pde_verbose=true,makePlot=true)
#h = solve_pde(ICparams,p,g)


"""
Define cost function to optimize (minimize)
"""
function costFun(ICparams::AbstractVector{Typ},p,g) where {Typ}

    # Compute C using my own ODE solver
    h=solve_pde(ICparams,p,g)

    # Compute cost (error)
    cost = (maximum(h) - 1e-3)^2

    return cost
end
# Test cost function 
#p,g,ICparams = probelmSetup(Ngrid=10,pde_verbose=true,makePlot=true)
#cost = costFun(ICparams,p,g)


# """
# Optimization - Newton's Method w/ various AD backends
# """
# function optimize_Own(fg!,k,tol; optim_verbose=false)
#     optim_verbose && println("\nSolving Optimization Problem with own optimizer")

#     # Set AD backend 
#     α=1e-3 #0.9 # Slow down convergence to compare methods
#     iter=0
#     converged = false
#     F = similar(k)
#     G = similar(k)
#     while converged == false
#         iter += 1

#         F = fg!(F,G,k)

#         # Update 
#         k -= α*G

#         # Limit k[1]
#         k[1] = max(eps(0.0),k[1])

#         # Check if converged
#         converged = (abs(F) < tol || iter == 500 || maximum(abs.(G)) < tol)

#         # Output for current IC
#         optim_verbose && @printf(" %5i, k = %15.5g, Cost Function = %15.6g, max(∇) = %15.6g \n",iter,k[1],F,maximum(abs.(G))) 
#     end

#     return k # Optimized IC
# end

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
    k = Optim.minimizer(optimize(Optim.only_fg!(fg!), k₀, BFGS(),
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

        # # Tape
        # results = DiffResults.GradientResult(k)
        # f_tape = ReverseDiff.GradientTape(f,k)
        # compiled_f_tape = ReverseDiff.compile(f_tape)
        # cfg = ReverseDiff.GradientConfig(k)
        # function fg_rev!(F,G,k)
        #     ReverseDiff.gradient!(results,compiled_f_tape,k)
        #     if F !== nothing
        #         F = DiffResults.value(results)
        #     end
        #     if G !== nothing 
        #         G[:] = DiffResults.gradient(results)
        #     end
        #     return F
        # end

        # Config
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
    p,g,k_guess = probelmSetup(Ngrid=5, pde_verbose=false, makePlot=false)
    f_for,fg_for! = optimSetup(k_guess, p, g, ADmethod="Forward")
    f_rev,fg_rev! = optimSetup(k_guess, p, g, ADmethod="Reverse")

    # Test computing value 
    #@time value = f(k_guess)
    #println("value = ",value)

    k_test=copy(k_guess)
    k_test += 1e-3rand(size(k_test,1))

    # Test computing value and gradient
    println("Calling fg! with k=",k_test)
    F_for=0.0; G_for=zeros(size(k_guess)); @time F_for = fg_for!(F_for,G_for,k_test)
    F_rev=0.0; G_rev=zeros(size(k_guess)); @time F_rev = fg_rev!(F_rev,G_rev,k_test)
    println("value - Forward = ",F_for)
    println("value - Reverse = ",F_rev)
    println(" grad - Forward = ",G_for)
    println(" grad - Reverse = ",G_rev)

    # Run Optimizers
    # k_Optim_for = optimize_Optim(fg_for!,k_guess,p.tol,optim_verbose=false);
    # k_Optim_rev = optimize_Optim(fg_rev!,k_guess,p.tol,optim_verbose=false);
    # k_Own   = optimize_Own(  fg!,k_guess,p.tol,optim_verbose=true)

    # println("Optim - Forward")
    # println(" -    optimum k = ",k_Optim_for)
    # println(" - f(k_optimum) = ",f_for(k_Optim_for))
    # plot_taylor(k_Optim_for,g.xm,g.ym)

    # println("Optim - Reverse")
    # println(" -    optimum k = ",k_Optim_rev)
    # println(" - f(k_optimum) = ",f_rev(k_Optim_rev))
    # plot_taylor(k_Optim_rev,g.xm,g.ym)
    
    # println("Own")
    # println(" -    optimum k = ",k_Own)
    # println(" - f(k_optimum) = ",f(k_Own))
    # plot_taylor(k_Own,g.xm,g.ym)

end
test_methods()

# # Time function & gradient evaluations
# function time_methods()

#     # Grids to test
#     grids = [10,20,40,80,100]

#     # Preallocate time arrays
#     teval = zeros(length(grids))
#     tgrad = zeros(length(grids))
#     tboth = zeros(length(grids))

#     # Initialize value and gradient arrays
#     F=[0.0,]
#     G=[0.0,]
    
#     # Iterate over grids
#     iter = 0
#     for grid in grids
#         iter += 1

#         # Setup problem for this grid
#         p,g,k_guess = probelmSetup(Ngrid=grid, pde_verbose=false)
#         f, g!, fg! = optimSetup([k_guess,], p, g, ADmethod="Forward")

#         # Test evaluating f
#         teval[iter] = @elapsed f(k_guess)
#         println("teval=",teval[1:iter])

#         # Test evaluating g!
#         tgrad[iter] = @elapsed g!(G,[k_guess,])
#         println("tgrad=",tgrad[1:iter])

#         # Test evaluating fg!
#         tboth[iter] = @elapsed fg!(F,G,[k_guess,])
#         println("tgrad=",tboth[1:iter])
#     end
# end
# #time_methods()

end # module