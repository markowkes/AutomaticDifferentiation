# Diffusion equation
# dC/dt = D d^2C/dx^2
using Plots
using Printf
using DifferentialEquations
using Zygote
using SciMLSensitivity # Needed for Zygote
using ForwardDiff
using ReverseDiff # Can't get this to work!!!
using FiniteDifferences

# Wrapper package that implements various AD backends
using AbstractDifferentiation 


```
Define RHS of PDE 
dC/dt = d/dx(D dC/dx)
```
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

```
Define function to differentiate
```
function diff_of_solution(D)
    _prob = remake(prob,p=[dx,D])
    sol=solve(_prob,Tsit5(),reltol=1e-6,abstol=1e-6,saveat=tfinal)
    diff=maximum(sol[end])-minimum(sol[end])
    return diff
end

```
Test various methods to compute derivatives
```
function testDerivative(f, D)
    println("Derivative using various methods")
    
    # Finite difference
    t = @elapsed begin
        delta = 1e-8
        f1 = f(D + delta)
        f2 = f(D - delta)
        derivative = (f1 - f2) / (2 * delta)
    end
    @printf("%20s: df/dD = %10.6g in %10.6g s\n","Finite Difference",derivative,t)
    
    # ForwardDiff
    t = @elapsed(derivative = ForwardDiff.derivative(f, D))
    @printf("%20s: df/dD = %10.6g in %10.6g s \n","ForwardDiff",derivative,t)

    # Zygote
    t = @elapsed(derivative = Zygote.gradient(f,D))
    @printf("%20s: df/dD = %10.6g in %10.6g s \n","Zygote",derivative[1],t)

    # ReverseDiff ## Not working
    #t = @elapsed(derivative = ReverseDiff.gradient(f, [D]))
    #@printf("%20s: df/dD = %10.6g in %10.6g s \n","ReverseDiff",derivative,t)

    # AD.ForwardDiff
    ab = AD.ForwardDiffBackend()
    t = @elapsed(derivative = AD.derivative(ab,f,D))
    @printf("%20s: df/dD = %10.6g in %10.6g s \n","AD.ForwardDiff",derivative[1],t)

    # AD.Zygote
    ab = AD.ZygoteBackend()
    t = @elapsed(derivative = AD.derivative(ab,f,D))
    @printf("%20s: df/dD = %10.6g in %10.6g s \n","AD.Zygote",derivative[1],t)

    # # AD.ReverseDiff ## Not working
    # ab = AD.ReverseDiffBackend()
    # t = @elapsed(derivative = AD.derivative(ab,f,D))
    # @printf("%20s: df/dD = %10.6g \n","AD.ReverseDiff",derivative[1],t)

    # AD.FiniteDifferences
    ab = AD.FiniteDifferencesBackend()
    t = @elapsed(derivative = AD.derivative(ab,f,D))
    @printf("%20s: df/dD = %10.6g in %10.6g s \n","AD.FiniteDifferences",derivative[1],t)

end

# ------------------
# Inputs
# ------------------
tfinal = 2
L = 2.0
Nx = 50
D=0.05

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
println("Solved successfully")
myplt = plot(sol(tfinal))
display(myplt)

# ------------------
# Test Derivatives
# ------------------
println("Testing derivative d(diff_of_solution)/dD(D=",D,")")
testDerivative(diff_of_solution,D)