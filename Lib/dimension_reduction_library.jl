################################################################################################
## Dimension Reduction Codes:
################################################################################################
# Library dependencies:
using LinearAlgebra
include("./optimization_functions_library.jl")

# RSE Dimesnion Reduction Parameters:
mutable struct dimRedParams
    λw :: Float64                       # Regularization parameter
    αw :: Float64                       # Step Size parameter
    d :: Int64                          # Dimension of the Embedding, reduced dimension
    threshold :: Float64                # Threshold, loss criterion
    noIteration :: Int64                # Number of Iterations, loss criterion
end

################################################################################################
## FUNCTIONS:
################################################################################################
## Dimension Reduction Codes (Robust Sparse Embedding):

#= FUNCTIONS FOR DIMENSION REDUCTION:
=#
function DimReductionKernelRSE(dataStack::Array{Matrix{T}}, params::dimRedParams) where {T<:Union{ComplexF64,Float64}}
    noData = length(dataStack);
    eye = Matrix{Float64}(1.0I, noData, noData);
    Y = Matrix{Union{ComplexF64,Float64}};
    Yhat::Array{Matrix{T}} = [];
    for data in dataStack
        # data = dataStack[:,:,i];
        
        # Learning Manifold:
        W = learnManifoldFeatureSpace(data, params);

        # Embedding in the lower dimension manifold (Compression):
        Aux = Matrix{ComplexF64}(I, size(W)) - W;
        Aux = Aux * Aux';
        Aux = 0.5 * (Aux + Aux');
        F = eigen(Aux, sortby=abs);
        d_dimred = F.vectors[:, 2:(params.d+1)];
        Y = Matrix(d_dimred');    
        # if i == 1
        #     Yhat = kron(eye[i, :], Y);
        # else
        #     Yhat = hcat(Yhat, kron(eye[i, :], Y));
        # end 
        push!(Yhat, Y);
    end
    return Yhat;
end

function learnManifoldFeatureSpace(X::Array{T,2}, p::dimRedParams, tau::Float64=0.0) where {T<:Union{ComplexF64, Float64, Float32}}
    # Hyperparameter initialization:
    Nk = size(X, 2);
    λw = p.λw; 
    αw = p.αw;
    count = 0;
    loss  = 1;

    # Constants required for computation to ease initialization:
    Iw = Matrix{T}(I, Nk, Nk);
    O = ones(T, 1, Nk);
    b = [1.0, 0.0];
    AinvB = zeros(T, Nk, Nk);
    AAinv = zeros(T, Nk, Nk, Nk);
    for i = 1:Nk
        A = [O; Iw[i, :]'];
        Ainv = pinv(A);
        AinvB[:, i] = Ainv*b;
        AAinv[:,:,i] = Iw - (A')*(Ainv');
    end

    # STEP 1: Lipschitz Coefficient and Learning Rate:
    
    XhX = X'*X;
    Lw = opnorm(XhX);
    λ = 1.98*(1 - αw)/Lw; 
    λλw = λ*λw;
    println("λλw ", λλw);

    # STEP 2: Initialiation:
    # H0 = similar(X);
    H0 = zeros(T, Nk, Nk);
    mul!(H0, pinv(X), X); 

    # STEP 3:
    TH0 = similar(H0);
    for i = 1:Nk
        TH0[:, i] = AAinv[:,:,i]*H0[:,i] - AinvB[:, i];
    end
    TaH0 = αw*TH0 + (1 - αw)*H0;

    # STEP 4:
    gradH0 = similar(H0);
    # gradH0 = - X + X*H0;
    gradH0 = XhX * H0 - XhX;

    H12 = TaH0 - λ * gradH0;

    # STEP 5:
    H1 = similar(H0);
    softThresholdingProximal!(H1, H12, λλw);
    TH1 = similar(TH0);

    # STEP 6:
    while (loss > p.threshold && count < p.noIteration)

    # STEP 7:
        # gradH1 = - X + X*H1;
        gradH1 = XhX * H1 - XhX + tau*(H1-H0);
        for i = 1:Nk
            TH1[:, i] = AAinv[:,:,i]*H1[:,i] - AinvB[:, i];
        end
        H12 = H12 + TH1 - TaH0 - λ * (gradH1 - gradH0);
        TaH0 = αw*TH1 + (1 - αw)*H1;
        softThresholdingProximal!(H1, H12, λλw);

    # STEP 8:
        loss = norm(H1 - H0, 2)/norm(H0, 2);
        # println("==== K-RSE Compression (Loop) : ", loss, ", Iteration : ", count);
        count = count + 1;
        H0 = copy(H1);
        gradH0 = copy(gradH1);
    end
    println("==== K-RSE Compression : ", loss, ", Iteration : ", count);
    return H1;
end