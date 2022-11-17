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
Determine number of basis functions needed for Nth order representation
"""
function Taylor_nBasis_order1(N)
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

# Legendre polynomials
# Note assumes x∈[-1,1]
function L(n,x)
    if n==0
        L=1
    elseif n==1
        L=x
    elseif n==2
        L=0.5*(3.0x^2-1.0)
    elseif n==3
        L=0.5*(5.0x^3-3.0x)
    elseif n==4
        L=0.125*(35.0x^4-30.0x^2+3.0)
    elseif n==5
        L=0.125*(63.0x^5-70.0x^3+15.0x)
    elseif n==6
        L=0.0625*(231.0x^6-315.0x^4+105.0x^3-35.0x)
    elseif n==7
        L=0.0625*(429.0x^7-693.0x^5+315.0x^3-35.0x)
    elseif n==8
        L=0.0078125*(6435.0x^8-12012.0x^6+6930.0x^4-1260.0x^2+35)
    else
        error("Legendre order ",n," not programmed")
    end

end

function taylor_Legendre(k,x,y)
t=(
     k[1]*L(0,x)
    +k[2]*L(2,x)
    +k[3]*L(2,y)
    +k[4]*L(4,x)
    +k[5]*L(2,x)*L(2,y)
    +k[6]*L(4,y)
    +k[7]*L(6,y)
    +k[8]*L(8,y)
)
end

function Taylor_nBasis_order(N)
    return 8
end

"""
Print first N basis functions
"""
function taylor_printBasis(N;k=ones(N))
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
        @printf(" Basis #%3i = %10.4g * x^%3i * y^%3i \n",n,k[n],xpow,ypow)
    end
    return nothing
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
    if eltype(k) <: ForwardDiff.Dual || eltype(k) <: ReverseDiff.TrackedReal
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