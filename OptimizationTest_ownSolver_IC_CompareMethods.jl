
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
function solve_pde(C::AbstractVector{T},p::param,g::grid,ADbackend) where {T}

    @unpack CFL,Nx,D,tfinal = p
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
    dt=CFL*dx^2/D

    # Number of time iterations
    nStep=ceil(tfinal/dt)

    # Recompute dt with this nStep
    dt=tfinal/nStep

    for iter in 1:nStep

        # Update time
        t = t + dt

        # Compute fluxes = D*dC/dx
        flux[1]=0.0  # No flux at boundaries
        flux[Nx+1]=0.0
        for i=2:Nx
            flux[i] = D*(C[i] - C[i-1]) / dx
        end
        
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
        cost += 1e-6(C₀[i]-mean(C₀))^2 # Add cost to large ICs
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
    α=0.9 # Slow down convergence to compare methods
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
Heaviside Function
"""
function heaviside(x)
    return map(x -> ifelse(x>=0.0,1.0,0.0),x)
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
        tol = 1e-10,
    )

    # Grid
    x = range(0.0, p.L, length=p.Nx+1)
    xm = 0.5 * (x[1:p.Nx] + x[2:p.Nx+1])
    dx = x[2] - x[1]
    g=grid(x=x,xm=xm,dx=dx)

    # Create goal for final solution 
    C₀=heaviside(g.xm .- 0.5) - heaviside(g.xm .- 1.5)

    # Solve PDE to create realistic C_goal
    C_goal=solve_pde(C₀,p,g,"Forward")

    # Initial guess 
    C₀_guess=ones(size(g.xm))

    return p,g,C₀,C₀_guess,C_goal
end

"""
Main Driver to test running various methods
"""
function testMethods()
    p,g,C₀,C₀_guess,C_goal = probelmSetup(Nx=20)

    # Optimize with own routine and various backends
    C₀_for = optimize_own(C₀_guess,C_goal,p,g,"Forward",verbose=true)
    C₀_rev = optimize_own(C₀_guess,C_goal,p,g,"Reverse",verbose=true)
    C₀_zyg = optimize_own(C₀_guess,C_goal,p,g,"Zygote",verbose=true)

    # Plot specified and optimized ICs
    myplt = plot( xm,C₀,label="Specified IC used to make C_goal")
    myplt = plot!(xm,C₀_for,markershape=:square,label="ForwardDiff")
    myplt = plot!(xm,C₀_rev,markershape=:circle,label="ReverseDiff")
    myplt = plot!(xm,C₀_zyg,linestyle=:dash,label="Zygote")
    myplt = plot!(title="Optimized Initial Condition")
    display(myplt)

    # Plot expected final solution (C_goal) 
    # and final solutions from optimized ICs
    myplt = plot( xm,C_goal,linewidth=2,label="C_goal")
    myplt = plot!(xm,solve_pde(C₀_for,p,g,"Forward"),markershape=:square,label="ForwarddDiff")
    myplt = plot!(xm,solve_pde(C₀_rev,p,g,"Reverse"),markershape=:circle,label="ReverseDiff")
    myplt = plot!(xm,solve_pde(C₀_zyg,p,g,"Zygote"),linestyle=:dash,label="Zygote")
    myplt = plot!(title="Final solution using optimized Initial Condition")
    display(myplt)
end

# Run method comparison
#println("Calling testMethods()"); @timeit to "testMethods" testMethods()

"""
Main Driver to do timing test
"""
function timeMethods()

    # Mesh sizes to test
    Nx=[5,10,20,30,40,60]

    # Preallocate timing array
    timing=zeros(3,length(Nx))

    # Loop over various mesh sizes 
    for n in eachindex(Nx)

        p,g,C₀,C₀_guess,C_goal = probelmSetup(Nx=Nx[n])

        # Optimize with own routine and various backends
        println("Calling timing test with Nx=",p.Nx)
        verbose=false
        timing[1,n] = timingTest(optimize_own,C₀_guess,C_goal,p,g,"Forward",verbose)
        timing[2,n] = timingTest(optimize_own,C₀_guess,C_goal,p,g,"Reverse",verbose)
        timing[3,n] = timingTest(optimize_own,C₀_guess,C_goal,p,g,"Zygote" ,verbose)
    end

    myplot = plot( Nx,timing[1,:],label="Forward")
    myplot = plot!(Nx,timing[2,:],label="Reverse")
    myplot = plot!(Nx,timing[3,:],label="Zygote")
    myplot = plot!(xlabel="# Grid Points")
    myplot = plot!(ylabel="Computation Time [s]")
    myplot = plot!(legend=:topleft)
    display(myplot)
        
end

"""
Runs function multiple times and returns median time 
""" 
function timingTest(fun,C₀_guess,C_goal,p,g,method,verbose)
    @printf("  - %10s",method)
    nTimes = 3
    times=zeros(nTimes)
    @timeit to method begin 
        for n in eachindex(times)
            @printf(" .")
            _, times[n] = @timed(fun(C₀_guess,C_goal,p,g,method,verbose=verbose))
        end
    end
    @printf(" %16.8g s \n",median(times))
    return median(times)
end
# Run method comparison
println("Calling timeMethods()"); timeMethods()

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
show(to)