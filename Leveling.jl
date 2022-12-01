module Leveling

#export probelmSetup, solve_pde, param, grid

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

""" 
Parameter structure
"""
@with_kw struct param
    CFL = 0.2 :: Float64
    Lx :: Float64
    Ly :: Float64
    Nx :: Int16
    Ny :: Int16
    tfinal :: Float64
    tol :: Float64
    pde_verbose :: Bool
    makePlot :: Bool
    outFreq :: Int16
end

""" 
Grid structure
"""
@with_kw struct grid
    x  :: Vector{Float64}
    y  :: Vector{Float64}
    xm :: Vector{Float64}
    ym :: Vector{Float64}
    dx :: Float64
    dy :: Float64
end

"""
Solve PDE with own ODE time integrator
"""
function solve_pde(p::param, g::grid, IC_h::Function, IC_c::Function, uParam::AbstractVector{Typ}) where {Typ}

    @unpack CFL, Nx, Ny, tfinal, pde_verbose, makePlot = p
    @unpack x, y, xm, ym, dx, dy = g

    M,c⁰,σₛ,σᵣ,σ⁰,L,H,ρ,grav,T,D⁰,E⁰,n = uParam

    # Initial condition
    t=0.0
    h=zeros(Typ,Nx,Ny)
    c=zeros(Typ,Nx,Ny)
    for i=1:Nx, j=1:Ny
        h[i,j] = IC_h(xm[i],ym[j])
        c[i,j] = IC_c(xm[i],ym[j])
    end
    
    
    # Determine timestep
    # CFL=0.2
    # dt=CFL*dx^2/max(maximum(kx),maximum(ky))
    dt=1e-5 #CFL*dx^2/1e-2

    # Number of time iterations
    nStep=ceil(tfinal/dt)

    # Recompute dt with this nStep
    dt=tfinal/nStep

    # Periodic boundary conditions helpers
    function per(i,j)
        iper = i; 
        while iper< 1; iper += Nx; end
        while iper>Nx; iper -= Nx; end
        jper = j; 
        while jper< 1; jper += Ny; end
        while jper>Ny; jper -= Ny; end
        return CartesianIndex(iper,jper)
    end

    # Get values for plot limits
    hmin = DualValue(minimum(h))
    hmax = DualValue(maximum(h))
    cmin = DualValue(minimum(c))
    cmax = DualValue(maximum(c))

    # RHS of PDEs 
    function rhs(h::AbstractArray{Typ},c::AbstractArray{Typ}) where {Typ}
        # Preallocate work arrays 
        rhs_h = zeros(Typ,Nx,Ny)
        rhs_c = zeros(Typ,Nx,Ny)
        Fx    = zeros(Typ,Nx+1,Ny)
        Fy    = zeros(Typ,Nx,Ny+1)
        
        # -------------------------
        #  Define useful functions
        # -------------------------

        # Material properties that depend on concentration
        μ(c) = 3*0.1 #exp( M * (c - c⁰) )           # Viscosity             ??? is this missing μ⁰
        D(c) = 1.0e-6 #T/L^2*D⁰*exp(-M*(c - c⁰))     # Diffusivity
        σ(c) = 0.03 #( σₛ + (σᵣ - σₛ) * c ) / σ⁰    # Surface tension
        E(c) = 1e-4*(1-c) #T/H*E⁰*(1-c)^n                # Evaporation Rate - f=1
        AΓ = 0.0 #( 3/2*L^2/H^2 )*( (σᵣ - σₛ)/σ⁰ )        # Importance of surface tension gradient
        B⁰ = ρ*grav*L^2/σ⁰                           # Bond Number

        # Interpolation to cell faces
        h_fx(i,j) = 0.5 * ( h[per(i-1,j)] + h[per(i,j)] ) # h interpolated to x face
        h_fy(i,j) = 0.5 * ( h[per(i,j-1)] + h[per(i,j)] ) # h interpolated to y face
        c_fx(i,j) = 0.5 * ( c[per(i-1,j)] + c[per(i,j)] ) # c interpolated to x face
        c_fy(i,j) = 0.5 * ( c[per(i,j-1)] + c[per(i,j)] ) # c interpolated to y face
        
        # Gradients on faces     
        dhdx_fx(i,j) = ( h[per(i,j)] - h[per(i-1,j)] ) / dx
        dhdy_fy(i,j) = ( h[per(i,j)] - h[per(i,j-1)] ) / dy
        dcdx_fx(i,j) = ( c[per(i,j)] - c[per(i-1,j)] ) / dx
        dcdy_fy(i,j) = ( c[per(i,j)] - c[per(i,j-1)] ) / dy

        # Gradients at cell centers
        dcdx(i,j) = ( c[per(i+1,j)] - c[per(i-1,j)] ) / 2dx
        dcdy(i,j) = ( c[per(i,j+1)] - c[per(i,j-1)] ) / 2dy 
        dhdx(i,j) = ( h[per(i+1,j)] - h[per(i-1,j)] ) / 2dx 
        dhdy(i,j) = ( h[per(i,j+1)] - h[per(i,j-1)] ) / 2dy 
        
        # Lapacian at cell center
        ∇²h(i,j) =                                  
                ( ( dhdx_fx(i+1,j) - dhdx_fx(i,j) ) /dx 
                + ( dhdy_fy(i,j+1) - dhdy_fy(i,j) ) /dy )

        # Gradient of Lapacian at cell faces
        d∇²hdx_fx(i,j) = ( ∇²h(i,j) - ∇²h(i-1,j) ) / dx
        d∇²hdy_fy(i,j) = ( ∇²h(i,j) - ∇²h(i,j-1) ) / dy

        # Gradient of Lapacian at cell center
        d∇²hdx(i,j) = ( ∇²h(i+1,j) - ∇²h(i-1,j) ) / 2dx
        d∇²hdy(i,j) = ( ∇²h(i,j+1) - ∇²h(i,j-1) ) / 2dy

        # --------------
        #  RHS of dh/dt
        # --------------
        # Fluxes on x faces
        for j=1:Ny, i=1:Nx+1
            hf = h_fx(i,j)
            cf = c_fx(i,j)
            Fx[i,j] = dy *
                ( AΓ * hf^2 / μ(cf) * dcdx_fx(i,j)      # Surface tension gradient 
                + hf^3 / μ(cf) * σ(cf) * d∇²hdx_fx(i,j) # Surface tension
                - B⁰ * hf^3 / μ(cf) * dhdx_fx(i,j)      # Gravity
                )
        end
        # Fluxes on y faces
        for j=1:Ny+1, i=1:Nx
            hf = h_fy(i,j)
            cf = c_fy(i,j)
            Fy[i,j] = dx *
                ( AΓ * hf^2 / μ(cf) * dcdy_fy(i,j)      # Surface tension gradient 
                + hf^3 / μ(cf) * σ(cf) * d∇²hdy_fy(i,j) # Surface tension
                - B⁰ * hf^3 / μ(cf) * dhdy_fy(i,j) )    # Gravity
        end
        # Compute RHS
        for j=1:Ny, i=1:Nx
            divg = 1.0/(dx*dy) *      # Divergence terms 
                ( Fx[i+1,j] - Fx[i,j] 
                + Fy[i,j+1] - Fy[i,j] 
            ) 
            rhs_h[i,j] = (
                - divg # Divergence terms (surf ten grad, surf ten, gravity)
                - E(c[i,j])   # Evaporation
            )
        end

        # --------------
        #  RHS of dc/dt
        # --------------
        # Fluxes on x faces
        for j=1:Ny, i=1:Nx+1
            cf = c_fx(i,j)
            Fx[i,j] = dy *
                ( D(cf) * h_fx(i,j) * dcdx_fx(i,j) ) # Diffusion
        end
        # Fluxes on y faces
        for j=1:Ny+1, i=1:Nx
            cf = c_fy(i,j)
            Fy[i,j] = dx *
                ( D(cf) * h_fy(i,j) * dcdy_fy(i,j) ) # Diffusion
        end
        # Velocity
        u(i,j) = 
            ( AΓ * h_fx(i,j) / μ(c_fx(i,j)) * dcdx(i,j) 
            + h_fx(i,j)^2 / μ(c_fx(i,j)) * d∇²hdx(i,j)
            - B⁰ * h_fx(i,j)^2 / μ(c_fx(i,j)) * dhdx(i,j) )
        v(i,j) = 
            ( AΓ * h_fy(i,j) / μ(c_fy(i,j)) * dcdy(i,j) 
            + h_fy(i,j)^2 / μ(c_fy(i,j)) * d∇²hdy(i,j)
            - B⁰ * h_fy(i,j)^2 / μ(c_fy(i,j)) * dhdy(i,j) )
        # Compute RHS
        for j=1:Ny , i=1:Nx
            divg = 1.0/(dx*dy) *  # Divergence terms 
                ( Fx[i+1,j] - Fx[i,j] 
                + Fy[i,j+1] - Fy[i,j] 
            ) 
            rhs_c[i,j] = (
                1.0/h[i,j] * divg   # Divergence terms (diffusion)
                + E(c[i,j])/h[i,j] * c[i,j] # Evaporation
                - u(i,j)*dcdx(i,j) - v(i,j)*dcdy(i,j)  # Convection  
            )
        end

        return rhs_h, rhs_c
    end

    for iter in 1:nStep

        # Update time
        t += dt

        # Update h & c - Euler
        rhs_h, rhs_c = rhs(h,c) 
        h += dt*rhs_h
        c += dt*rhs_c

        # # 4ᵗʰ-order Runge-Kutta
        # k1_h, k1_c = rhs(h           ,c           )
        # k2_h, k2_c = rhs(h+0.5dt*k1_h,c+0.5dt*k1_c) 
        # k3_h, k3_c = rhs(h+0.5dt*k2_h,c+0.5dt*k2_c) 
        # k4_h, k4_c = rhs(h+   dt*k3_h,c+   dt*k3_c)
        # rhs_h = 1.0/6.0*(k1_h + 2.0k2_h + 2.0k3_h + k4_h)
        # rhs_c = 1.0/6.0*(k1_c + 2.0k2_c + 2.0k3_c + k4_c)
        # h += dt*rhs_h
        # c += dt*rhs_c

        # Outputs
        if pde_verbose
            if rem(iter,p.outFreq)==0
                @printf("iter = %5i t = %6.3g max(rhs_h) = %6.3g max(rhs_c) = %6.3g\n",
                    iter,t,DualValue(maximum(abs.(rhs_h))),DualValue(maximum(abs.(rhs_c))))
                if makePlot
                    if Ny>1
                        p1 = plot(xm,ym,DualValue(h)',
                            st=:surface,
                            title=@sprintf("Time = %6.3g",t),
                            xlabel = "x",
                            ylabel = "y",
                            zlabel = "h(x,y)",
                            zlim=(0,hmax))
                        p2 = plot(xm,ym,DualValue(c)',
                            st=:surface,
                            xlabel = "x",
                            ylabel = "y",
                            zlabel = "c(x,y)",
                            zlim=(0,1))
                        p3 = plot(p1, p2, layout = (2,1) )
                        display(p3)
                    else
                        p1 = plot(xm,DualValue(h[:,1]),label="h(x,t)",
                            title=@sprintf("Time = %6.3g",t),
                            xlabel = "x",
                            ylabel = "h(x,y)",
                            ylim=(0,hmax),
                            legend = false)
                        p2 = plot(xm,DualValue(c[:,1]),label="c(x,t)",
                            xlabel = "x",
                            ylabel = "c(x,y)",
                            ylim=(0,1),
                            legend = false)
                        p3 = plot(p1, p2, layout = (2,1) )
                        display(p3)
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
    # Parameters
    p=param(
        Lx = 0.004, #0.5,
        Ly = 0.004/Ngrid, #0.5/Ngrid,
        Nx = Ngrid,
        Ny = 1,
        tfinal = 2.0,
        tol = 1e-5,
        pde_verbose = pde_verbose,
        makePlot = makePlot, # Requires pde_verbose to also be true
        outFreq = 10000,
    )

    # Grid
    x = range(0.0, p.Lx, length=p.Nx+1)
    y = range(0.0, p.Ly, length=p.Ny+1)
    xm = 0.5 * (x[1:p.Nx] + x[2:p.Nx+1])
    ym = 0.5 * (y[1:p.Ny] + y[2:p.Ny+1])
    dx = x[2] - x[1]
    dy = y[2] - y[1]
    g=grid(x=x,y=y,xm=xm,ym=ym,dx=dx,dy=dy)

    # Initial condition
    IC_h(x,y) = 125e-6 .+ 50e-6*cos.(2π*x/p.Lx)
    IC_c(x,y) = 0.55

    # Initial guess for uncertain parameters
    M = 15.0
    c⁰ = 0.55
    σₛ = 23.375
    σᵣ = 30.875
    σ⁰ = 27.5
    L = 0.5 # Same as Lx=Ly???
    H = 0.0056 # Same as h⁰???
    ρ = 0.77
    grav = 9.8 # ???
    T = 1.0 # ???
    D⁰ = 1.0e-5
    E⁰ = 0.2e-6
    n = 1.0
    uParam = [M,c⁰,σₛ,σᵣ,σ⁰,L,H,ρ,grav,T,D⁰,E⁰,n]

    return p,g,IC_h,IC_c,uParam
end
# Test running solver
#`p,g,IC_h,IC_c,uParam = probelmSetup(Ngrid=10,pde_verbose=true,makePlot=true)
#h,c = solve_pde(p,g,IC_h,IC_c,uParam)


"""
Define cost function to optimize (minimize)
"""
function costFun(uParam::AbstractVector{Typ},p,g,IC_h,IC_c) where {Typ}

    # Compute C using my own ODE solver
    h,c = solve_pde(p,g,IC_h,IC_c,uParam)

    # Compute cost (error)
    cost = sum((h.-1e-3).^2)

    return cost
end
# Test cost function 
# p,g,IC_h,IC_c,uParam = probelmSetup(Ngrid=10,pde_verbose=true,makePlot=true)
# cost = costFun(uParam,p,g,IC_h,IC_c)
# println("Cost = ",cost)

"""
Optimization - Optim.jl
"""
function optimize_Optim(fg!,uParam,tol; optim_verbose=false)
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
    k = Optim.minimizer(optimize(Optim.only_fg!(fg!), uParam, BFGS(),
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
function optimSetup(uParam,p,g,IC_h,IC_c; ADmethod="Forward",chunk=50)
    # Function value
    f = uParam -> costFun(uParam,p,g,IC_h,IC_c)

    # Choose method based on ADmethod input
    if ADmethod == "Forward"
        # Value and Gradient
        results = DiffResults.GradientResult(uParam)
        tag = ForwardDiff.Tag(f, eltype(uParam))
        cfg = ForwardDiff.GradientConfig(f, uParam, ForwardDiff.Chunk{min(chunk,prod(size(uParam)))}(), tag)
        function fg_for!(F,G,uParam)
            ForwardDiff.gradient!(results,f,uParam,cfg)
            if F !== nothing
                F = DiffResults.value(results)
            end
            if G !== nothing 
                G[:] = DiffResults.gradient(results)
            end
            return F
        end
        fg! = (F,G,uParam) -> fg_for!(F,G,uParam)

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
        results = DiffResults.GradientResult(uParam)
        cfg = ReverseDiff.GradientConfig(uParam)
        function fg_rev!(F,G,uParam)
            ReverseDiff.gradient!(results, f, uParam, cfg)
            if F !== nothing
                F = DiffResults.value(results)
            end
            if G !== nothing 
                G[:] = DiffResults.gradient(results)
            end
            return F
        end

        fg! = (F,G,uParam) -> fg_rev!(F,G,uParam)
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
    p,g,IC_h,IC_c,uParam_guess = probelmSetup(Ngrid=10, pde_verbose=false, makePlot=false)
    f_for,fg_for! = optimSetup(uParam_guess,p,g,IC_h,IC_c, ADmethod="Forward")
    #f_rev,fg_rev! = optimSetup(uParam_guess,p,g,IC_h,IC_c, ADmethod="Reverse")

    # Test computing value 
    #@time value = f(k_guess)
    #println("value = ",value)

    uParam_test=copy(uParam_guess)
    #k_test += 1e-3rand(size(k_test,1))

    # Test computing value and gradient
    println("Calling fg! with uParam=",uParam_test)
    F_for=0.0; G_for=zeros(size(uParam_guess)); @time F_for = fg_for!(F_for,G_for,uParam_test)
    #F_rev=0.0; G_rev=zeros(size(uParam_guess)); @time F_rev = fg_rev!(F_rev,G_rev,uParam_test)
    println("value - Forward = ",F_for)
    #println("value - Reverse = ",F_rev)
    println(" grad - Forward = ",G_for)
    #println(" grad - Reverse = ",G_rev)

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

# # # Time function & gradient evaluations
# # function time_methods()

# #     # Grids to test
# #     grids = [10,20,40,80,100]

# #     # Preallocate time arrays
# #     teval = zeros(length(grids))
# #     tgrad = zeros(length(grids))
# #     tboth = zeros(length(grids))

# #     # Initialize value and gradient arrays
# #     F=[0.0,]
# #     G=[0.0,]
    
# #     # Iterate over grids
# #     iter = 0
# #     for grid in grids
# #         iter += 1

# #         # Setup problem for this grid
# #         p,g,k_guess = probelmSetup(Ngrid=grid, pde_verbose=false)
# #         f, g!, fg! = optimSetup([k_guess,], p, g, ADmethod="Forward")

# #         # Test evaluating f
# #         teval[iter] = @elapsed f(k_guess)
# #         println("teval=",teval[1:iter])

# #         # Test evaluating g!
# #         tgrad[iter] = @elapsed g!(G,[k_guess,])
# #         println("tgrad=",tgrad[1:iter])

# #         # Test evaluating fg!
# #         tboth[iter] = @elapsed fg!(F,G,[k_guess,])
# #         println("tgrad=",tboth[1:iter])
# #     end
# # end
# # #time_methods()

end # module