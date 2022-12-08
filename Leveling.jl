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
    L  :: Float64
    H  :: Float64
    Nx :: Int16
    Ny :: Int16
    dt :: Float64
    tfinal :: Float64
    maxSteps :: Int64
    tol :: Float64
    pde_verbose :: Bool
    makePlot :: Bool
    outFreq :: Int16
    outputName :: String
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
function solve_pde(p::param, g::grid, uParam::AbstractVector{Typ}) where {Typ}

    @unpack L, H, CFL, Nx, Ny, dt, maxSteps, tfinal, pde_verbose, makePlot = p
    @unpack x, y, xm, ym, dx, dy = g

    σ̃ₛ,σ̃ᵣ,σ̃⁰,μ̃⁰,D̃⁰,ρ,Ẽ⁰,c⁰,n,M,h̃⁰,ã⁰,λ̃x,λ̃y,grav = uParam

    # Initial condition
    t=0.0
    h=zeros(Typ,Nx,Ny)
    c=zeros(Typ,Nx,Ny)
    for i=1:Nx, j=1:Ny
        h[i,j] = (h̃⁰ + ã⁰*cos(2π*xm[i]*L/λ̃x)*cos(2π*ym[j]*L/λ̃y))/H
        c[i,j] = c⁰
    end
    
    # Determine timestep
    # CFL=0.2
    # dt=CFL*dx^2/max(maximum(kx),maximum(ky))
    #dt=1e-4 #CFL*dx^2/1e-2

    # Number of time iterations
    nStep=ceil(tfinal/dt)

    # Recompute dt with this nStep
    #dt=tfinal/nStep

    # Periodic boundary conditions helper
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
    function rhs(t,h::AbstractArray{Typ},c::AbstractArray{Typ}) where {Typ}
        # Preallocate work arrays 
        rhs_h = zeros(Typ,Nx,Ny)
        rhs_c = zeros(Typ,Nx,Ny)
        Fx    = zeros(Typ,Nx+1,Ny)
        Fy    = zeros(Typ,Nx,Ny+1)
        
        # -------------------------
        #  Define useful functions
        # -------------------------

        # Material properties that depend on concentration
        # All non-dimensional
        T = 3.0μ̃⁰*L^4/(σ̃⁰*H^3)                      # Characteristic timescale
        B⁰ = ρ*grav*L^2/σ̃⁰                           # Bond Number
        μ(c) = exp( M * (c - c⁰) )            # Viscosityμ⁰
        D(c) = T/L^2*D̃⁰*exp(-M*(c - c⁰))     # Diffusivity
        σ(c) = ( σ̃ₛ + (σ̃ᵣ - σ̃ₛ) * c ) / σ̃⁰    # Surface tension
        f(x,y,t)=1.0 # Spatial and temporal variations in evaporation rate
        E(c,x,y,t) = T/H*Ẽ⁰*f(x,y,t)*(max.(0.0,1.0-c))^n                # Evaporation Rate - 
        AΓ = ( 3/2*L^2/H^2 )*((σ̃ᵣ - σ̃ₛ)/σ̃⁰)        # Importance of surface tension gradient

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

        # Velocity
        u(i,j) = 
            ( AΓ * h_fx(i,j) / μ(c_fx(i,j)) * dcdx_fx(i,j)  # Surface tension gradient 
            + h_fx(i,j)^2 / μ(c_fx(i,j)) * σ(c_fx(i,j)) * d∇²hdx_fx(i,j) # Surface tension
            - B⁰ * h_fx(i,j)^2 / μ(c_fx(i,j)) * dhdx_fx(i,j) )  # Gravity
        v(i,j) = 
            ( AΓ * h_fy(i,j) / μ(c_fy(i,j)) * dcdy_fy(i,j)  # Surface tension gradient
            + h_fy(i,j)^2 / μ(c_fy(i,j)) * σ(c_fy(i,j)) * d∇²hdy(i,j) # Surface tension
            - B⁰ * h_fy(i,j)^2 / μ(c_fy(i,j)) * dhdy_fy(i,j) )  # Gravity 

        # --------------
        #  RHS of dh/dt
        # --------------
        # Fluxes on x faces
        for j=1:Ny, i=1:Nx+1
            Fx[i,j] = dy * u(i,j)*h_fx(i,j)
        end
        # Fluxes on y faces
        for j=1:Ny+1, i=1:Nx
            Fy[i,j] = dx * v(i,j)*h_fy(i,j)
        end
        # Compute RHS
        for j=1:Ny, i=1:Nx
            divg = 1.0/(dx*dy) * ( Fx[i+1,j] - Fx[i,j] + Fy[i,j+1] - Fy[i,j] ) 
            rhs_h[i,j] = (
                - divg # Divergence terms (surf ten grad, surf ten, gravity)
                #- E(c[i,j],xm[i],ym[j],t)   # Evaporation
            )
        end

        # --------------
        #  RHS of dc/dt
        # --------------
        # Fluxes on x faces
        for j=1:Ny, i=1:Nx+1
            Fx[i,j] = dy * ( D(c_fx(i,j)) * h_fx(i,j) * dcdx_fx(i,j) ) # Diffusion
        end
        # Fluxes on y faces
        for j=1:Ny+1, i=1:Nx
            Fy[i,j] = dx * ( D(c_fy(i,j)) * h_fy(i,j) * dcdy_fy(i,j) ) # Diffusion
        end
        # Compute RHS
        for j=1:Ny , i=1:Nx
            divg = 1.0/(dx*dy) * ( Fx[i+1,j] - Fx[i,j] + Fy[i,j+1] - Fy[i,j] ) 
            rhs_c[i,j] = (
                1.0/h[i,j] * divg   # Divergence terms (diffusion)
                + E(c[i,j],xm[i],ym[j],t)/h[i,j] * c[i,j] # Evaporation
                - u(i,j)*dcdx(i,j) - v(i,j)*dcdy(i,j)  # Convection  
            )
        end

        return rhs_h, rhs_c
    end

    anim = Animation()
    for iter in 1:min(nStep,maxSteps)

        # Update time
        t += dt

        # Update h & c - Euler
        rhs_h, rhs_c = rhs(t,h,c) 
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
                        #plotly()
                        gr()
                        ir=1:Int(Nx/2)
                        jr=1:Int(Ny/2)
                        p1 = plot(xm[ir],ym[jr],DualValue(h[ir,jr])',
                            st=:surface,
                            title=@sprintf("Time = %6.3g",t),
                            xlabel = "x",
                            ylabel = "y",
                            zlabel = "h(x,y)",
                            zlim=(0,hmax),
                            )
                        p2 = plot(xm[ir],ym[jr],DualValue(c[ir,jr])',
                            st=:surface,
                            xlabel = "x",
                            ylabel = "y",
                            zlabel = "c(x,y)",
                            #zlim=(0,1),
                            )
                        p3 = plot(p1, p2, layout = (2,1), size = (600,800) )
                        display(p3)
                        frame(anim,p3)
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
                        frame(anim,p3)
                    end 
                end
            end
        end
        
    end

    return h,c,anim
end

"""
Setup problem to test
"""
function probelmSetup(;case="2D", Ngrid=50, pde_verbose=false, makePlot=false)

    if case == "2D"
        # Initial guess for uncertain parameters (2D values)
        σ̃ₛ = 23.375
        σ̃ᵣ = 30.875
        σ̃⁰ = 27.5
        μ̃⁰ = 15.9
        D̃⁰ = 1.0e-5
        ρ = 0.77
        Ẽ⁰ = 0.2e-6
        c⁰ = 0.55
        n = 1.0
        M = 15.0
        h̃⁰ = 0.0056
        ã⁰ = 0.0025 
        λ̃x = 0.4
        λ̃y = Inf
        grav = 980 # cm/s^2
    elseif case =="3D"
        # Initial guess for uncertain parameters (3D values)
        σ̃ₛ = 23.0
        σ̃ᵣ = 30.0
        σ̃⁰ = 26.5
        μ̃⁰ = 5.0
        D̃⁰ = 1.0e-5
        ρ = 0.75
        Ẽ⁰ = 1.0e-6
        c⁰ = 0.5
        n =  0.5
        M = 15.0
        h̃⁰ = 0.005
        ã⁰ = 0.0025 
        λ̃x = 1.0
        λ̃y = 1.0
        grav = 980 # cm/s^2
    else
        error("Unknown case!")
    end

    # Package parameters
    uParam = [σ̃ₛ,σ̃ᵣ,σ̃⁰,μ̃⁰,D̃⁰,ρ,Ẽ⁰,c⁰,n,M,h̃⁰,ã⁰,λ̃x,λ̃y,grav]

    # Parameters
    if case == "2D"
        Ngrid=50
        # Domain size
        Lx = 2.0
        Ly = 2.0/Ngrid
        # Grid points
        Nx = Ngrid
        Ny = 1
        # Simulation time (non-dimensional)
        dt = 1e-7
        tfinal = 5e-2
        maxSteps = 1e10
        outFreq = 10000
        outputName = "Leveling"
    elseif case == "3D"
        Ngrid=40
        # Domain size
        Lx = 2.0 
        Ly = 2.0
        # Grid points
        Nx = Ngrid
        Ny = Ngrid
        # Simulation time (non-dimensional)
        dt = 5e-8
        tfinal = 2e-2
        maxSteps = 100000
        outFreq = 1000
        outputName = "Leveling3D"
    else
        error("Unknown case!")
    end

    p = param(
        Lx = Lx,
        Ly = Ly,
        Nx = Nx,
        Ny = Ny,
        dt = dt,
        tfinal = tfinal,
        maxSteps = maxSteps,
        # Characteristic length  and coating height
        L = λ̃x/2.0,
        H = h̃⁰,
        tol = 1e-5,
        pde_verbose = pde_verbose,
        makePlot = makePlot, # Requires pde_verbose to also be true
        outFreq = outFreq,
        outputName = outputName,
    )

    # Grid
    x = range(0.0, p.Lx, length=p.Nx+1) # Non-dimensional grid
    y = range(0.0, p.Ly, length=p.Ny+1)
    xm = 0.5 * (x[1:p.Nx] + x[2:p.Nx+1])
    ym = 0.5 * (y[1:p.Ny] + y[2:p.Ny+1])
    dx = x[2] - x[1] # Non-dimensional grid size
    dy = y[2] - y[1]
    g=grid(x=x,y=y,xm=xm,ym=ym,dx=dx,dy=dy)

    return p,g,uParam
end
# Test running solver
p,g,uParam = probelmSetup(case="2D",pde_verbose=true,makePlot=true)
h,c,anim = solve_pde(p,g,uParam)
if anim.frames != []
    gif(anim,p.outputName*".gif")
end

# """
# Define cost function to optimize (minimize)
# """
# function costFun(uParam::AbstractVector{Typ},p,g,IC_h,IC_c) where {Typ}

#     # Compute C using my own ODE solver
#     h,c = solve_pde(p,g,IC_h,IC_c,uParam)

#     # Compute cost (error)
#     cost = sum((h.-1e-3).^2)

#     return cost
# end
# # Test cost function 
# # p,g,IC_h,IC_c,uParam = probelmSetup(Ngrid=10,pde_verbose=true,makePlot=true)
# # cost = costFun(uParam,p,g,IC_h,IC_c)
# # println("Cost = ",cost)

# """
# Optimization - Optim.jl
# """
# function optimize_Optim(fg!,uParam,tol; optim_verbose=false)
#     optim_verbose && println("\nSolving Optimization Problem with Optim and AD")
#     # lower =-Inf*ones(size(k₀))
#     # lower[1] = eps(0.0)
#     # upper = Inf*ones(size(k₀))
#     # k = Optim.minimizer(optimize(Optim.only_fg!(fg!), lower, upper, k₀, Fminbox(GradientDescent()),
#     #     Optim.Options(
#     #         g_tol = tol,
#     #         iterations = 100,
#     #         store_trace = false,
#     #         show_trace = optim_verbose,
#     #         #extended_trace = true,
#     #         )))
#     k = Optim.minimizer(optimize(Optim.only_fg!(fg!), uParam, BFGS(),
#             Optim.Options(
#                 g_tol = tol,
#                 iterations = 100,
#                 store_trace = false,
#                 show_trace = optim_verbose,
#                 #extended_trace = true,
#                 )))
#     return k # Optimized IC
# end

# """
# Setup function and gradients to use Optimizers
# """
# function optimSetup(uParam,p,g,IC_h,IC_c; ADmethod="Forward",chunk=50)
#     # Function value
#     f = uParam -> costFun(uParam,p,g,IC_h,IC_c)

#     # Choose method based on ADmethod input
#     if ADmethod == "Forward"
#         # Value and Gradient
#         results = DiffResults.GradientResult(uParam)
#         tag = ForwardDiff.Tag(f, eltype(uParam))
#         cfg = ForwardDiff.GradientConfig(f, uParam, ForwardDiff.Chunk{min(chunk,prod(size(uParam)))}(), tag)
#         function fg_for!(F,G,uParam)
#             ForwardDiff.gradient!(results,f,uParam,cfg)
#             if F !== nothing
#                 F = DiffResults.value(results)
#             end
#             if G !== nothing 
#                 G[:] = DiffResults.gradient(results)
#             end
#             return F
#         end
#         fg! = (F,G,uParam) -> fg_for!(F,G,uParam)

#     elseif ADmethod == "Reverse"
#         # ReverseDiff Gradient

#         # # Tape
#         # results = DiffResults.GradientResult(k)
#         # f_tape = ReverseDiff.GradientTape(f,k)
#         # compiled_f_tape = ReverseDiff.compile(f_tape)
#         # cfg = ReverseDiff.GradientConfig(k)
#         # function fg_rev!(F,G,k)
#         #     ReverseDiff.gradient!(results,compiled_f_tape,k)
#         #     if F !== nothing
#         #         F = DiffResults.value(results)
#         #     end
#         #     if G !== nothing 
#         #         G[:] = DiffResults.gradient(results)
#         #     end
#         #     return F
#         # end

#         # Config
#         results = DiffResults.GradientResult(uParam)
#         cfg = ReverseDiff.GradientConfig(uParam)
#         function fg_rev!(F,G,uParam)
#             ReverseDiff.gradient!(results, f, uParam, cfg)
#             if F !== nothing
#                 F = DiffResults.value(results)
#             end
#             if G !== nothing 
#                 G[:] = DiffResults.gradient(results)
#             end
#             return F
#         end

#         fg! = (F,G,uParam) -> fg_rev!(F,G,uParam)
#     else
#         error("Unknown ADmethod")
#     end

#     return f,fg!
# end 

# # Test running the PDE solver
# function test_methods()
#     #p,g,k_guess = probelmSetup(Ngrid=20, pde_verbose=true)
#     #T = solve_pde(k_guess,p,g)

#     # Setup Optimization problem
#     p,g,IC_h,IC_c,uParam_guess = probelmSetup(Ngrid=10, pde_verbose=false, makePlot=false)
#     f_for,fg_for! = optimSetup(uParam_guess,p,g,IC_h,IC_c, ADmethod="Forward")
#     #f_rev,fg_rev! = optimSetup(uParam_guess,p,g,IC_h,IC_c, ADmethod="Reverse")

#     # Test computing value 
#     #@time value = f(k_guess)
#     #println("value = ",value)

#     uParam_test=copy(uParam_guess)
#     #k_test += 1e-3rand(size(k_test,1))

#     # Test computing value and gradient
#     println("Calling fg! with uParam=",uParam_test)
#     F_for=0.0; G_for=zeros(size(uParam_guess)); @time F_for = fg_for!(F_for,G_for,uParam_test)
#     #F_rev=0.0; G_rev=zeros(size(uParam_guess)); @time F_rev = fg_rev!(F_rev,G_rev,uParam_test)
#     println("value - Forward = ",F_for)
#     #println("value - Reverse = ",F_rev)
#     println(" grad - Forward = ",G_for)
#     #println(" grad - Reverse = ",G_rev)

#     # Run Optimizers
#     # k_Optim_for = optimize_Optim(fg_for!,k_guess,p.tol,optim_verbose=false);
#     # k_Optim_rev = optimize_Optim(fg_rev!,k_guess,p.tol,optim_verbose=false);

#     # println("Optim - Forward")
#     # println(" -    optimum k = ",k_Optim_for)
#     # println(" - f(k_optimum) = ",f_for(k_Optim_for))
#     # plot_taylor(k_Optim_for,g.xm,g.ym)

#     # println("Optim - Reverse")
#     # println(" -    optimum k = ",k_Optim_rev)
#     # println(" - f(k_optimum) = ",f_rev(k_Optim_rev))
#     # plot_taylor(k_Optim_rev,g.xm,g.ym)
    
# end
# #test_methods()

end # module