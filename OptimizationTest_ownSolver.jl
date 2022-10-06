module OptimizationTest 

    using Plots
    using Printf
    using DifferentialEquations
    using ForwardDiff
    using AbstractDifferentiation 
    using Optim

    """
    Own ODE time integrator 
    """
    function solve_ode(D::T) where {T}

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
        C = exp.(-(xm .- L / 2.0) .^ 2 / 0.1)
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

    # ------------------
    # Test PDE Solver 
    # ------------------
    println("Solving ODE")
    xm,C=solve_ode(0.01)
    myplt = plot(xm,C)
    display(myplt)
    println("Solved successfully")

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
        xm,C=solve_ode(p[1])

        # Cost
        maxmin_goal=0.2
        maxmin_solv=maximum(C)-minimum(C)
        return (maxmin_goal - maxmin_solv)^2
    end

    # Plot cost function 
    Ds=0.01:0.02:0.6
    Fs=similar(Ds)
    for i in 1:length(Ds)
        Fs[i]=costFun(Ds[i])
    end
    myplt = plot(Ds,Fs)
    xlabel!("D")
    ylabel!("costFun(D)")
    display(myplt)

    # ------------------------------
    # Optimization - Newton's Method
    # D = D - dC/dD / d^2C/dD^2
    # ------------------------------
    function optimize_newton(D)
        println("\nSolving Optimization Problem with Newton's Method and AD")
        ab = AD.ForwardDiffBackend()
        f=1.0
        iter=0
        while f>1e-16
            iter += 1
            #df=ForwardDiff.gradient(costFun,[D,])
            #println("df=",df[1])
            (f, df, ddf) = AD.value_gradient_and_hessian(ab,costFun,[D,])
            df = df[1][1]
            ddf=ddf[1][1]
            D -=  df/ddf
            @printf(" %5i, D=%20.16g, f=%15.6g , df=%15.6g , ddf=%10.6g \n",iter,D,f,df,ddf)
        end
    end
    optimize_newton(0.01)

    # ----------------------
    # Optimization - Optim.jl
    # ----------------------
    function optimize_Optim(D)
        println("\nSolving Optimization Problem with Optim and AD")

        # Uses automatic differentiation and Newton's Method
        od = TwiceDifferentiable(costFun, [D,]; autodiff = :forward)
        Dopt = Optim.minimizer(optimize(od, [D,], Newton()))
        println("Optimum D = ",Dopt[1])
    end
    optimize_Optim(0.01)

end