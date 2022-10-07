module OptimizationTest 

    using Plots
    using Printf
    using DifferentialEquations
    using ForwardDiff
    using AbstractDifferentiation 
    using Optim
    using Statistics

    """
    Own ODE time integrator 
    """
    function solve_ode(D::T,sigma::T) where {T}

        # Inputs
        tfinal = 2.0
        L = 2.0
        Nx = 50

        # Grid
        x = range(0.0, L, length=Nx + 1)
        xm = 0.5 * (x[1:Nx] + x[2:Nx+1])
        dx = x[2] - x[1]

        # Preallocate
        C   =zeros(T,Nx)
        dC  =zeros(T,Nx)
        flux=zeros(T,Nx+1)

        # Initial condition
        C .= exp.(-(xm .- L / 2.0) .^ 2 / sigma)
        t = 0.0
        
        # Determine timestep
        CFL=0.5
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
            C += dt * dC
            
        end

        return xm,C
    end

    # ------------------------------
    # ------------------------------
    #  Optimization
    # ------------------------------
    # ------------------------------
    """
    Define function to optimize (minimize)
    """
    function costFun(p)
        # Compute C using my own ODE solver
        xm,C=solve_ode(p[1],p[2])

        # Cost
        maxmin_goal=0.2
        D_goal=0.3
        maxmin_solv=maximum(C)-minimum(C)
        return (maxmin_goal - maxmin_solv)^2 + (D_goal - p[1])^2
    end

    # Plot cost function 
    function plot_costFunction()
        plotly()
        D=0.01:0.002:0.6
        σ=0.01:0.002:0.6
        f=zeros(length(D),length(σ))
        f_min=1e20
        i_min=0; j_min=0
        for i in 1:length(D), j in 1:length(σ)
            f[i,j]=costFun([D[i],σ[j]])
            if f[i,j] < f_min 
                i_min = i 
                j_min = j
                f_min = f[i,j]
            end
        end
        println("Minimum: costFunction(D=",D[i_min],",σ=",σ[j_min],") = ",f[i_min,j_min])
        myplt = surface(D,σ,f)
        xlabel!("D")
        ylabel!("σ")
        zlabel!("costFun(D,σ)")
        display(myplt)    
    end
    plot_costFunction()

    # ------------------------------
    # Optimization - Newton's Method
    # D = D - dC/dD / d^2C/dD^2
    # ------------------------------
    function optimize_newton(D,sigma)
        println("\nSolving Optimization Problem with Newton's Method and AD")
        # Set AD backend to ForwardDiff 
        ab = AD.ForwardDiffBackend()
        f=1.0
        iter=0
        p=[D,sigma]
        α=1.0
        while f>1e-16
            iter += 1
            # Compute derivatives using AD
            (f, Grad, Hess) = AD.value_gradient_and_hessian(ab,costFun,p)
            # Update p 
            p -= α*Hess[1]\Grad[1]
            @printf(" %5i, D=%20.16g, σ=%20.16g, f=%15.6g \n",iter,p[1],p[2],f)
        end
    end
    #optimize_newton(0.01,0.1)

    # # ----------------------
    # # Optimization - Optim.jl
    # # ----------------------
    # function optimize_Optim(D,sigma)
    #     println("\nSolving Optimization Problem with Optim and AD")

    #     # Uses automatic differentiation and Newton's Method
    #     od = TwiceDifferentiable(costFun, [D,sigma]; autodiff = :forward)
    #     Dopt = Optim.minimizer(optimize(od, [D,sigma], Newton()))
    #     println("Optimum D = ",Dopt[1])
    # end
    # optimize_Optim(0.01,0.1)

end