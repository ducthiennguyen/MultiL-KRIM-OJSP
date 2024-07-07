################################################################################################
## Optimization Functions: (Gradient and Proximal Algorithms)
# Library Dependencies:
using LinearAlgebra

function affineMappingRight(Y::Matrix{T}) where {T<:Union{ComplexF64, Float64}}
    # Each column adds up to 1
    (Nr, Nc) = size(Y);
    Ones = ones(T, Nr);
    return Y - (1/Nr) .* Ones * (Ones' * Y - ones(T, Nc)');
end

function affineMappingLeft(Y::Matrix{T}) where {T<:Union{ComplexF64, Float64}}
    # Each row adds up to 1
    (Nr, Nc) = size(Y);
    Ones = ones(T, Nc);
    return Y - (1/Nc) .* (Y * Ones - ones(T, Nr)) * Ones';
end

#=
FUNCTION: Proximal Operator fot l1-norm minimization:
INPUT:  X = Matrix, type Real or Complex 
        lambda = thresholding paramtere
OUTPUT: Y = Soft Thresholded Output Matrix
=#
function softThresholdingProximal!(Y::Matrix{T}, X::Matrix{T}, λ::Float64) where {T<:Union{ComplexF64, Float64, Float32}} 
    Y .= @. X * (1 - λ/max(abs(X), λ));     # [Y]_ij = [X]_ij (1 - λ/max(|[X]_ij|, λ)); 
end

#=
FUNCTION: Proximal Operator fot l1-norm minimization:
INPUT:  X = Matrix, type Real or Complex
        lambda = thresholding paramtere
OUTPUT: Y = Soft Thresholded Output Matrix
=#
function boundConstraintProximal!(Y::Matrix{T}, X::Matrix{T}, Cx::Float64) where {T<:Union{ComplexF64,Float64}}
    for i = 1:size(X, 2)
        Y[:,i] .= X[:,i] * (Cx/max(Cx, norm(X[:,i])));
    end
    return Y;
end

function grad_k1!(G, P, K2, PK2, K1, n, m, kparams, type="gaussian")
    """
    Gradient of k_{1;n,m} at the point A, in case of exponential kernel
    Inplace calculation on G
    P: matrix
    K2: kernel matrix from Landmarks
    """
    gamma = kparams.alpha;
    r = kparams.degree;
    c = kparams.intercept;
    if type == "gaussian"
        G .= (-2*gamma * K1[n, m] * (conj(K2[n, m]) - K2[n, n]) * PK2);
    elseif type == "polynomial"
        G .= (r * (dot(PK2[:, n], PK2[:, m]) + c) ^ (r-1) * K2[m, n] * PK2);
    end
end

################################################################################################
