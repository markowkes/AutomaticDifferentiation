using Plots
using Printf
using DifferentialEquations
using Zygote
using SciMLSensitivity # Needed for Zygote
using ForwardDiff
using ReverseDiff # Can't get this to work!!!
using AbstractDifferentiation 

"""
Define RHS of PDE 
dC/dt = d/dx(D dC/dx)
"""
function rhs!(dC,C,p,t)

    dx,D=p

    # # Compute fluxes = D*dC/dx - interior faces
    flux=similar(C,Nx+1)
    flux[1]=0.0
    flux[Nx+1]=0.0
    for i=2:Nx
        flux[i] = D*(C[i] - C[i-1]) / dx
    end
    # Compute RHS dC/dt
    for i in 1:Nx
        dC[i]=(flux[i+1] - flux[i]) / dx
    end

    return dC
end

# ------------------
# Inputs
# ------------------
tfinal = 2
L = 2.0
Nx = 50
D=0.01

# Grid
x = range(0.0, L, length=Nx + 1)
xm = 0.5 * (x[1:Nx] + x[2:Nx+1])
dx = x[2] - x[1]

# Initial condition
C0 = exp.(-(xm .- L / 2.0) .^ 2 / 0.1)

# ------------------
# Test PDE Solver 
# ------------------
println("Solving ODE")
prob = ODEProblem(rhs!,C0,(0,tfinal),[dx,D])
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
myplt = plot(sol(tfinal))
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
    _prob = remake(prob,p=[dx,p[1]])
    sol=solve(_prob,Tsit5(),reltol=1e-6,abstol=1e-6,saveat=tfinal)
    maxmin_goal=0.4
    maxmin_solv=maximum(sol[end])-minimum(sol[end])
    return (maxmin_goal - maxmin_solv)^2
end

# Plot cost function 
Ds=0.0:0.05:0.6
Fs=similar(Ds)
for i in 1:length(Ds)
    Fs[i]=costFun(Ds[i])
end
myplt = plot(Ds,Fs)
xlabel!("D")
ylabel!("costFun(D)")
display(myplt)
println("Setup Optimization Problem")

"""
Optimization - Newton's Method
"""
function optimize_newton(D)
    ab = AD.ForwardDiffBackend()
    for iter in 1:10
        # Compute derivatives using AD
        (f, df, ddf) = AD.value_gradient_and_hessian(ab,costFun,[D,])
        df = df[1][1]
        ddf=ddf[1][1]    
        D -=  df/ddf
        @printf(" %5i, D=%10.6g, f=%10.6g , df=%10.6g , ddf=%10.6g \n",iter,D,costFun([D,]),df,ddf)
    end
end
println("Solving Optimization Problem with Newton's Method")
optimize_newton(D)


using Optim
"""
Optimization - Optim.jl
"""
function optimize_optim(D)
    # Using fintie differences (unstable)
    #results = Optim.minimizer(optimize(costFun,[D,],BFGS()))

    # Uses automatic differentiation and Newton's Method
    od = TwiceDifferentiable(costFun, [D,]; autodiff = :forward)
    Dopt = Optim.minimizer(optimize(od, [D,], Newton()))
    println("Optimum D = ",Dopt[1])
end
println("Solving Optimization Problem with Optim.jl")
optimize_optim(D)

