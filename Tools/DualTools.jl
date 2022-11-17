"""
Print an array that is possibly a Dual
"""
function printDual(Ain::AbstractMatrix,Name)
    println("---------------------------------------------")
    println("Printing ",Name)

    # Transpose of A to deal with x-y convention 
    A=Ain'

    # Size of A
    Nrow=size(A,1)
    Ncol=size(A,2)

    # Deal with different types of A
    if eltype(A) <: ForwardDiff.Dual || eltype(k) <: ReverseDiff.TrackedReal
        # Print values 
        println("values = ")
        values = map(A -> A.value, A)
        for row in Nrow:-1:1
            @printf("     ")
            for col in 1:Ncol
                @printf("%10.8g, ",values[row,col])
            end
            @printf("\n")
        end
        # Print partials
        Npartials = length(A[1].partials)
        partials = map(A -> A.partials, A)
        for partial in 1:Npartials
            @printf("partial #%i = \n",partial)
            for row in Nrow:-1:1
                @printf("     ")
                for col in 1:Ncol
                    # println(row,col,partial)
                    # println(partials[row,col][partial])
                    @printf("%10.8g, ",partials[row,col][partial])
                end
                @printf("\n")
            end
        end
    else
        for row = Nrow:-1:1
            @printf("     ")
            for col in 1:Ncol
                @printf("%10.8g, ",A[row,col])
            end
            @printf("\n")
        end
    end
    println("---------------------------------------------")
    return nothing
end

function printDual(A::Real,Name)
    if eltype(A) <: ForwardDiff.Dual || eltype(A) <: ReverseDiff.TrackedReal
        println(Name,".value   = ",A.value,"  ",A.partials)
    else
        println(Name," = ",A)
    end
    return nothing
end

function DualValue(A::Real)
    if eltype(A) <: ForwardDiff.Dual || eltype(A) <: ReverseDiff.TrackedReal
        return A.value
    else
        return A
    end
end

function printDualValue(Ain::AbstractMatrix,Name)
    println("---------------------------------------------")
    println("Printing ",Name)

    # Transpose of A to deal with x-y convention 
    A=Ain'

    # Size of A
    Nrow=size(A,1)
    Ncol=size(A,2)

    # Deal with different types of A
    if eltype(A) <: ForwardDiff.Dual || eltype(k) <: ReverseDiff.TrackedReal
        # Print values 
        println("values = ")
        values = map(A -> A.value, A)
        for row in Nrow:-1:1
            @printf("     ")
            for col in 1:Ncol
                @printf("%10.8g, ",values[row,col])
            end
            @printf("\n")
        end
    else
        for row = Nrow:-1:1
            @printf("     ")
            for col in 1:Ncol
                @printf("%10.8g, ",A[row,col])
            end
            @printf("\n")
        end
    end
    println("---------------------------------------------")
    return nothing
end