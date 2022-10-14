
####
# Optimizes initial condition to diffusion problem with
# - own PDE solver
# - Optimizers
#   - Newton's Method (own optimizer)
#   - Optim.jl (ForwardDiff & ReverseDiff)
###
module OptimizationTest

using Plots
using Printf
using DifferentialEquations
using ForwardDiff
using ReverseDiff
using Zygote
using AbstractDifferentiation 
using Optim
using Statistics
using UnPack
using Parameters
using BenchmarkTools
using TimerOutputs
using Printf

const to = TimerOutput()

""" 
Parameter structure
"""
@with_kw struct param
    CFL = 0.2
    L
    Nx
    tfinal
    D
    u
    tol
end

""" 
Grid structure
"""
@with_kw struct grid
    x
    xm
    dx
end


"""
Solve PDE with own ODE time integrator - Input is C₀ (IC)
"""
function solve_pde(C::AbstractVector{T},p::param,g::grid,ADbackend; makePlot=false) where {T}

    @unpack CFL,Nx,D,u,tfinal = p
    @unpack dx = g

    # Preallocate
    if ADbackend == "Forward" || ADbackend == "Reverse"
        dC  =zeros(T,Nx)
        flux=zeros(T,Nx+1)
    elseif ADbackend == "Zygote"
        # Preallocate
        dC  =Zygote.Buffer(C,Nx)
        Cout=Zygote.Buffer(C,Nx)
        flux=Zygote.Buffer(C,Nx+1)
    else
        error("Unknown ADbackend: ",ADbackend)
    end

    # Initial condition
    t = 0.0
    
    # Determine timestep
    CFL=0.2
    dt=CFL*min(dx^2/D,dx/u)

    # Number of time iterations
    nStep=ceil(tfinal/dt)

    # Recompute dt with this nStep
    dt=tfinal/nStep

    for iter in 1:nStep

        # Update time
        t = t + dt

        # Compute fluxes = D*dC/dx
        for i=2:Nx
            flux[i] = ( 
                D*(C[i] - C[i-1]) / dx # diffusion
                - u*C[i-1] # Advection
            )
        end
        # Periodic BCs
        flux[1] = ( 
            D*(C[1] - C[Nx]) / dx # diffusion
            - u*C[Nx] # Advection
        )
        flux[Nx+1] = flux[1]
        
        # Compute RHS dC/dt
        for i in 1:Nx
            dC[i]=(flux[i+1] - flux[i]) / dx
        end

        # Update C
        if ADbackend == "Forward" || ADbackend == "Reverse"
            C += dt * dC
        elseif ADbackend == "Zygote"
            for i in 1:Nx
                Cout[i] = C[i] + dt * dC[i]
            end
            C=Cout
        else
            error("Unknown ADbackend: ",ADbackend)
        end

        if makePlot
            myplt = plot(g.xm,C)
            display(myplt)
        end
        
    end

    # Postprocessing
    if ADbackend == "Forward" || ADbackend == "Reverse"
        # Nothing
    elseif ADbackend == "Zygote"
        C = copy(C)
    else
        error("Unknown ADbackend: ",ADbackend)
    end
    return C
end

"""
Define cost function to optimize (minimize)
"""
function costFun(C₀,C_goal,p,g,ADbackend)
    # Compute C using my own ODE solver
    C=solve_pde(C₀,p,g,ADbackend)

    # Compute cost (error)
    cost=0.0
    for i in eachindex(C)
        cost += ( C_goal[i] - C[i] )^2
        #cost += 1e-6(C₀[i]-mean(C₀))^2 # Add cost to large ICs
    end
    return cost
end

"""
Optimization - Newton's Method w/ various AD backends
"""
function optimize_own(C₀,C_goal,p,g,ADbackend; verbose=false, chunk=10)
    verbose && println("\nSolving Optimization Problem with Newton's Method and ",ADbackend)

    @unpack tol=p

    # Preprocessing
    if ADbackend == "Forward"
        @timeit to "HessianResults" results = DiffResults.HessianResult(C₀)
        @timeit to "Tag" tag = ForwardDiff.Tag((C₀ -> costFun(C₀,C_goal,p,g,ADbackend),ForwardDiff.hessian), eltype(C₀))
        @timeit to "Config" cfg = ForwardDiff.HessianConfig(C₀ -> costFun(C₀,C_goal,p,g,ADbackend),results,C₀,ForwardDiff.Chunk{min(chunk,length(C₀))}(), tag)

    elseif ADbackend == "Reverse"
        # Prepare results to compute value, gradient, and Hessian
        @timeit to "HessianResults" results = DiffResults.HessianResult(C₀)
        # Record and compile a tape of costFun()
        @timeit to "Tape" costFun_tape = ReverseDiff.HessianTape(C₀ -> costFun(C₀,C_goal,p,g,ADbackend),C₀)
        @timeit to "Compile Tape" compiled_costFun_tape = ReverseDiff.compile(costFun_tape)

    elseif ADbackend == "Zygote"
        # nothing
    else
        error("Unknown ADbackend: ",ADbackend)
    end

    # Set AD backend 
    α=1.0 #0.9 # Slow down convergence to compare methods
    myplt = plot(C₀,label="Initial condition")
    iter=0
    converged = false
    while converged == false
        iter += 1

        if ADbackend == "Forward"
            # Compute derivatives with config
            @timeit to "hessian!" ForwardDiff.hessian!(results,C₀ -> costFun(C₀,C_goal,p,g,ADbackend),C₀,cfg)
            # Extract value, gradient, and Hessian
            @timeit to "extract" f    = DiffResults.value(results)
            @timeit to "extract" Grad = DiffResults.gradient(results)
            @timeit to "extract" Hess = DiffResults.hessian(results)
            # Update C
            @timeit to "update" Cₙ = C₀ - α*(Hess\Grad)
        elseif ADbackend == "Reverse"
            # Run tape to compute results
            @timeit to "hessian!" results = ReverseDiff.hessian!(results,compiled_costFun_tape,C₀)
            # Extract value, gradient, and Hessian
            @timeit to "extract" f    = DiffResults.value(results)
            @timeit to "extract" Grad = DiffResults.gradient(results)
            @timeit to "extract" Hess = DiffResults.hessian(results)
            # Update C
            @timeit to "update" Cₙ = C₀ - α*(Hess\Grad)
        elseif ADbackend == "Zygote"
            # Evaluate function value, gradient, and Hessian - maybe there's a better way ...
            @timeit to "value" f=costFun(C₀,C_goal,p,g,ADbackend)
            @timeit to "Grad" Grad = Zygote.gradient(C₀ -> costFun(C₀,C_goal,p,g,ADbackend),C₀) # Reverse
            @timeit to "Hess" Hess = Zygote.hessian( C₀ -> costFun(C₀,C_goal,p,g,ADbackend),C₀) # Forward
            @timeit to "Update" Cₙ = C₀ - α*(Hess\Grad[1])
        else
            error("Unknown ADbackend: ",ADbackend)
        end

        @timeit to "postprocessing" begin        
            # Check if converged
            converged = (f < tol || iter == 500 || maximum(abs.(Grad[1])) < tol)

            # Transfer solution 
            C₀ = Cₙ

            # Output for current IC
            verbose && @printf(" %5i, Cost Function = %15.6g, max(∇) = %15.6g \n",iter,f,maximum(abs.(Grad[1]))) 
        end
    end

    return C₀ # Optimized IC
end


"""
Optimization - Optim.jl
"""

function optimize_Optim(f,g!,C₀,tol; verbose=false, chunk=10)
    verbose && println("\nSolving Optimization Problem with Optim and AD")
    Copt = Optim.minimizer(optimize(f, g!, C₀, BFGS(),
        Optim.Options(
            g_tol = tol,
            iterations = 1000,
            store_trace = false,
            show_trace = verbose,
            )))
    return Copt # Optimized IC
end

""" 
Heaviside Function
"""
function heaviside(x)
    return map(x -> ifelse(x==0.0,0.5,ifelse(x>=0.0,1.0,0.0)),x)
end


"""
Setup problem to test
"""
function probelmSetup(; Nx)
    # Inputs
    p=param(
        tfinal = 2.0,
        L = 2.0,
        Nx = Nx,
        D=0.1,
        u=0.75,
        tol = 1e-12,
    )

    # Grid
    x = range(0.0, p.L, length=p.Nx+1)
    xm = 0.5 * (x[1:p.Nx] + x[2:p.Nx+1])
    dx = x[2] - x[1]
    g=grid(x=x,xm=xm,dx=dx)

    # Create goal for final solution 
    C_goal=sin.(2*pi*xm/p.L)+cos.(2*pi*xm/p.L) # Periodic, smooth function
    #C_goal=sin.(pi*xm/p.L) # Smooth function, not periodic -> discontinuity
    #C_goal=heaviside(g.xm .- 0.5) - heaviside(g.xm .- 1.5)

    # Solve PDE to create realistic C_goal
    #

    # Initial guess 
    C₀_guess=ones(size(g.xm))

    return p,g,C₀_guess,C_goal
end

"""
Setup function and gradients to use Optim.jl
"""
function optimSetup(C_goal,p,g)
    f_for = C₀ -> costFun(C₀,C_goal,p,g,"Forward")
    f_rev = C₀ -> costFun(C₀,C_goal,p,g,"Reverse")
    function g_for!(G,C₀) 
        G[:] = ForwardDiff.gradient(f_for,C₀)
    end
    function g_rev!(G,C₀) 
        G[:] = ReverseDiff.gradient(f_rev,C₀)
    end
    return f_for, f_rev, g_for!, g_rev!
end 

"""
Main Driver to test running various methods
"""
function testMethods()
    p,g,C₀_guess,C_goal = probelmSetup(Nx=30)

    # Optimize with own routine and various backends
    # C₀_for = optimize_own(C₀_guess,C_goal,p,g,"Forward",verbose=true)
    # C₀_rev = optimize_own(C₀_guess,C_goal,p,g,"Reverse",verbose=true)
    # C₀_zyg = optimize_own(C₀_guess,C_goal,p,g,"Zygote",verbose=true)

    # Optimize with Optim.jl
    f_for, f_rev, g_for!, g_rev! = optimSetup(C_goal,p,g)
    C₀_optim_for = optimize_Optim(f_for,g_for!,C₀_guess,p.tol,verbose=true)
    C₀_optim_rev = optimize_Optim(f_rev,g_rev!,C₀_guess,p.tol,verbose=true)

    # # Plot specified and optimized ICs
    myplt = plot()
    #myplt = plot( g.xm,C₀,label="Specified IC used to make C_goal")
    # myplt = plot!(g.xm,C₀_for,markershape=:square,label="ForwardDiff")
    # myplt = plot!(g.xm,C₀_rev,markershape=:circle,label="ReverseDiff")
    # myplt = plot!(g.xm,C₀_zyg,linestyle=:dash,label="Zygote")
    myplt = plot!(g.xm,C₀_optim_for,markershape=:square,label="Optim.jl ForwardDiff")
    myplt = plot!(g.xm,C₀_optim_rev,markershape=:circle,label="Optim.jl ReverseDiff")
    myplt = plot!(title="Optimized Initial Condition")
    myplt = plot!(legend = :outertopright)
    myplt = plot!(size=(1200,800))
    display(myplt)

    # Plot expected final solution (C_goal) 
    # and final solutions from optimized ICs
    myplt = plot()
    myplt = plot!( g.xm,C_goal,linewidth=2,label="C_goal")
    # myplt = plot!(g.xm,solve_pde(C₀_for,p,g,"Forward"),markershape=:square,label="ForwarddDiff")
    # myplt = plot!(g.xm,solve_pde(C₀_rev,p,g,"Reverse"),markershape=:circle,label="ReverseDiff")
    # myplt = plot!(g.xm,solve_pde(C₀_zyg,p,g,"Zygote"),linestyle=:dash,label="Zygote")
    myplt = plot!(g.xm,solve_pde(C₀_optim_for,p,g,"Forward"),markershape=:square,label="Optim.jl ForwardDiff")
    myplt = plot!(g.xm,solve_pde(C₀_optim_rev,p,g,"Reverse"),markershape=:circle,label="Optim.jl ReverseDiff")
    myplt = plot!(title="Final solution using optimized Initial Condition")
    myplt = plot!(legend = :outertopright)
    myplt = plot!(size=(1200,800))
    display(myplt)

    # 
end

# Run method comparison
println("Calling testMethods()"); testMethods()

"""
Main Driver to do timing test
"""
function timeMethods()

    # Mesh sizes to test
    Nx=[5,10,20,30,40]

    # Methods
    method =["Forward","Reverse","Zygote","Optim - Forward","Optim - Reverse"]

    # Preallocate timing array
    timing=zeros(5,length(Nx))

    # Loop over various mesh sizes 
    for n in eachindex(Nx)

        p,g,C₀,C₀_guess,C_goal = probelmSetup(Nx=Nx[n])
        f_for,f_rev,g_for!,g_rev! = optimSetup(C_goal,p,g)

        verbose = false
        fun = Array{Function}(undef,5)
        fun[1] = () -> optimize_own(C₀_guess,C_goal,p,g,"Forward",verbose=verbose)
        fun[2] = () -> optimize_own(C₀_guess,C_goal,p,g,"Reverse",verbose=verbose)
        fun[3] = () -> optimize_own(C₀_guess,C_goal,p,g,"Zygote", verbose=verbose)
        fun[4] = () -> optimize_Optim(f_for,g_for!,C₀_guess,p.tol,verbose=verbose)
        fun[5] = () -> optimize_Optim(f_rev,g_rev!,C₀_guess,p.tol,verbose=verbose)

        # Optimize with own routine and various backends
        println("Calling timing test with Nx=",p.Nx)
        for i in eachindex(fun)
            timing[i,n] = timingTest(fun[i],method[i])
        end
    end
    myplot = plot(title="Comparison of Computational Cost")
    for i in eachindex(method)
        myplot = plot!(Nx,timing[i,:],label=method[i])
    end
    myplot = plot!(xlabel="# Grid Points")
    myplot = plot!(ylabel="Computation Time [s]")
    myplot = plot!(legend=:topleft)
    display(myplot)
        
end

"""
Runs function multiple times and returns median time 
""" 
function timingTest(fun,method)
    @printf("  - %20s",method)
    nTimes = 3
    times=zeros(nTimes)
    @timeit to method begin 
        for n in eachindex(times)
            @printf(" .")
            _, times[n] = @timed(fun())
        end
    end
    @printf(" %16.8g s \n",median(times))
    return median(times)
end
# Run method comparison
# println("Calling timeMethods()"); timeMethods()

"""
Timing parts of solver
"""
function timeParts()
    # Setup Test
    p,g,C₀,C₀_guess,C_goal = probelmSetup(Nx=20)

    # Optimize with own routine and various backends
    println("Calling timing test with Nx=",p.Nx)
    @timeit to "Forward" optimize_own(C₀_guess,C_goal,p,g,"Forward",verbose=true)
    @timeit to "Reverse" optimize_own(C₀_guess,C_goal,p,g,"Reverse",verbose=true)
    @timeit to "Zygote"  optimize_own(C₀_guess,C_goal,p,g,"Zygote", verbose=true)

end
# println("Calling timeParts()"); timeParts()

""" 
Test chunck size using with ForwarddDiff
"""
function testChunk()

    p,g,C₀,C₀_guess,C_goal = probelmSetup(Nx=40)

    chunks = [5,10,20] # Should be a factor of Nx 
    for chunk in chunks
        @timeit to @printf("Chunk %3i",chunk) begin
            optimize_own(C₀_guess,C_goal,p,g,"Forward",verbose=true,chunk=chunk)
        end
    end
end
#println("Calling testChunk()"); testChunk()

# Show timer results
#show(to)

end