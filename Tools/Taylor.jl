using Printf

"""
Evaluate function represented by a 
2D Taylor series with coefficients k
"""
function taylor(k,x,y)
    t=0.0
    xpow = 0
    ypow = 0
    order = -1
    for n in eachindex(k)
        xpow -= 1
        ypow += 1
        if xpow == -1
            order += 1
            xpow = order
            ypow = 0
        end
        t += k[n] * x^xpow * y^ypow
    end
    return t
end

"""
Print first N basis functions
"""
function taylor_printBasis(N)
    xpow = 0
    ypow = 0
    order = -1
    for n in 1:N
        xpow -= 1
        ypow += 1
        if xpow == -1
            order += 1
            xpow = order
            ypow = 0
            println("Order ",order)
        end
        @printf(" Basis #%3i = x^%3i * y^%3i \n",n,xpow,ypow)
    end
    return nothing
end

"""
Determine number of basis functions needed for Nth order representation
"""
function Taylor_nBasis_order(N)
    n = 0
    xpow = 0
    ypow = 0
    order = -1
    while true
        n += 1
        xpow -= 1
        ypow += 1
        if xpow == -1
            order += 1
            xpow = order
            ypow = 0
            # Check if reached N
            if order == N+1
                return n-1
            end
        end
    end
end

""" 
Plot a Taylor series with coefficients k 
on a grid with points x,y
"""
function plot_taylor(k,x,y)
    Nx=length(x)
    Ny=length(y)
    f=zeros(Nx,Ny)
    # Deal with Duals
    if typeof(k[1]) <: ForwardDiff.Dual
        ks = map(k -> k.value,k)
    else
        ks = k 
    end
    # Compute function
    for i in eachindex(x), j in eachindex(y)
        f[i,j]=taylor(ks,x[i],y[j])
    end
    myplot = plot(x,y,f',
        st=:surface,
        #camera=(-30,30),
        xlabel = "x",
        ylabel = "y",
        zlabel = "f(x,y)",
        title  = @sprintf("min/max = %10.4g, %10.4g",minimum(f),maximum(f))
        )
    display(myplot)
    return nothing
end