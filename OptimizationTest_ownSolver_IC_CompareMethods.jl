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
    Optimization - Gradient Descent & Newton's Method
    """
    function optimize_own(C₀,C_goal,p,g,ADbackend)
        println("\nSolving Optimization Problem with Newton's Method and ",ADbackend)

        @unpack tol=p

        # Preprocessing
        if ADbackend == "Forward"
            ab = AD.ForwardDiffBackend()    
        elseif ADbackend == "Reverse"
            ab = AD.ReverseDiffBackend()
        elseif ADbackend == "Zygote"
            # nothing
        else
            error("Unknown ADbackend: ",ADbackend)
        end

        # Set AD backend 
        α=1
        myplt = plot(C₀,label="Initial condition")
        iter=0
        converged = false
        while converged == false
            iter += 1

            if ADbackend == "Forward" || ADbackend == "Reverse"
                (f, Grad, Hess) = AD.value_gradient_and_hessian(ab,C₀ -> costFun(C₀,C_goal,p,g,ADbackend),C₀)
                Cₙ = C₀ - α*(Hess[1]\Grad[1])
            elseif ADbackend == "Zygote"
                f=costFun(C₀,C_goal,p,g,ADbackend)
                Grad = Zygote.gradient(C₀ -> costFun(C₀,C_goal,p,g,ADbackend),C₀) # Reverse
                Hess = Zygote.hessian( C₀ -> costFun(C₀,C_goal,p,g,ADbackend),C₀) # Forward
                Cₙ = C₀ - α*(Hess\Grad[1])
            else
                error("Unknown ADbackend: ",ADbackend)
            end

            # Check if converged
            converged = (f < tol || iter == 500 || maximum(abs.(Grad[1])) < tol)

            # Transfer solution 
            C₀ = Cₙ

            # Output for current IC
            @printf(" %5i, Cost Function = %15.6g, max(∇) = %15.6g \n",iter,f,maximum(abs.(Grad[1]))) 
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
    Main Driver
    """
    function compareMethods()
        # Inputs
        p=param(
            tfinal = 2.0,
            L = 2.0,
            Nx = 20,
            D=0.1,
            tol = 1e-5,
        )

        # Grid
        x = range(0.0, p.L, length=p.Nx+1)
        xm = 0.5 * (x[1:p.Nx] + x[2:p.Nx+1])
        dx = x[2] - x[1]
        g=grid(x=x,xm=xm,dx=dx)
    

        # Create goal for final solution 
        ## Option 1
        #sigma=0.1; C₀=exp.(-(xm .- L / 2.0) .^ 2 / sigma) .+ 1.0
        ## Option 2
        C₀=heaviside(g.xm .- 0.5) - heaviside(g.xm .- 1.5)
        
        # Solve PDE to create realistic C_goal
        C_goal=solve_pde(C₀,p,g,"Forward")

        # Initial guess 
        C₀_guess=ones(size(g.xm))

        # Optimize with own routine and various backends
        C₀_for = optimize_own(C₀_guess,C_goal,p,g,"Forward")
        C₀_rev = optimize_own(C₀_guess,C_goal,p,g,"Reverse")
        C₀_zyg = optimize_own(C₀_guess,C_goal,p,g,"Zygote")

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
    compareMethods()
end

