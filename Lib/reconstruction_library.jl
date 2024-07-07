################################################################################################
## Reconstruction Tasks:
################################################################################################
## Library Dependencies:
using SparseArrays, MatrixEquations
using Random, LinearAlgebra, FFTW, ToeplitzMatrices

# Recovery Task Parameters:
mutable struct OptimizerParams
    N :: Int64                         # Phase Encoding Lines
    T :: Int64                         # Phase Frequency Lines
    # Nfr :: Int64                        # Number of Frames
    ζ :: Float64                        # Decay Rate parameter for Main Reconstruction Task
    γ :: Float64                        # Step Size parameter for Main Reconstruction Task
    λ1 :: Array{Float64}                # Regularization parameter 
    λ2 :: Float64                       # Regularization parameter
    λ3 :: Float64                       # Regularization parameter
    λs :: Float64                       # Regularization parameter
    λn :: Float64                       # Regularization parameter for nuclear norm
    μn :: Float64                       # Regularization parameter for nuclear norm
    λL :: Float64                       # Regularization parameter
    ϵ :: Float64                       # Regularization parameter
    C :: Array{Float64}                 # Bounding Constant parameter on Matrix A
    alpha :: Array{Float64}                 # Bounding Constant parameter on Matrix A
    CA :: Float64                       # Bounding Constant parameter on Matrix A
    CG :: Float64                       # Bounding Constant parameter on Matrix G
    τ :: Array{Float64}                 # Sub Optimisation Tasks parameter in general
    τA :: Float64                       # Sub Optimization Tasks Parameter for || A - An ||
    τG :: Float64                       # Sub Optimization Tasks Parameter for || G - Gn ||
    τB :: Float64                       # Sub Optimization Tasks Parameter for || B - Bn ||
    τS :: Float64                       # Sub Optimization Tasks Parameter for || B - Bn ||
    α :: Float64                        # Step Size Parameter for Sub Optimization Tasks in general
    αA :: Float64                       # Step Size Parameter for Sub Optimization Tasks
    αG :: Float64                       # Step Size Parameter for Sub Optimization Tasks
    αB :: Float64                       # Step Size Parameter for Sub Optimization Tasks
    αS :: Float64                       # Step Size Parameter for Sub Optimization Tasks
    noK :: Int64                        # Number of kernels employed
    threshold :: Float64                # Threshold of loss criterion for the Main Reconstruction Task                
    noIteration :: Int64                # Itertion Count Cut off in convergence criteria is 
                                        #    not met for the Main Reconstruction Task
    thresholdInner :: Float64           # Threshold of loss criterion for the Sub Optimization Tasks
    noIterationInner :: Int64           # Itertion Count Cut off in convergence criteria is 
                                        #    not met for the Sub Optimization Tasks
end

function MultiLKRIM(
    Y::Array{T1, 2},
    KL,
    Lap,
    Diff::Array{T1, 2},
    M::BitArray{2},
    Misses::BitArray{2},
    params::OptimizerParams,
    ImageData::Array{T1}) where {T1 <:Union{Float64, ComplexF64}}
    ## Matrix Size initialization
    rng = MersenneTwister(seed);
    N = params.N;
    T = params.T;
    (Nl1, Nl2) = size(KL);
    if size(ImageData, 1) != 1  
        img = reshape(ImageData, N, T);
        imgLoss::Array{Float64} = [];
        flag = 0;
    else
        flag = 1;
    end

    ## Reconstructions Hyperparameter Initialization:
    loss = 1e0;
    noIter = 0;
    γ = params.γ;
    ζ = params.ζ;

    # Parameter, Matrix Values to save computation in time in loops:
    local Mc = .!M;
    
    ## STEP 1: Low-resolution:
    local Xlow = Y;

    ## STEP 2: Random Initialiation:
    # X0:
    local Xn = copy(Xlow);
    local Xhat = copy(Xlow);
    local Yhat = copy(Xlow);
    # Left of kernel matrix
    local An = 1e-1*randn(rng, Float64, N, d*params.noK);
    local Dn = randn(rng, Float64, 2, 4);
    Dn = sparse(BlockDiagonal(repeat([Dn], params.noK))); # (2*params.noK, d*params.noK)
    local En = randn(rng, Float64, 6, d);
    En = sparse(BlockDiagonal(repeat([En], params.noK))); # (2*params.noK, d*params.noK)
    local Gn = randn(rng, Float64, d, Int(Nl1/params.noK));
    Gn = sparse(BlockDiagonal(repeat([Gn], params.noK))); # (d*params.noK, Nl1)
    
    # Right of kernel matrix
    local Bn = 1e-1*randn(rng, Float64, Nl2, T);

    # Others:
    local Xhat_prev = copy(Xn);
    local Xhat_new = similar(Xn);
    local temp = copy(Xn);
    local Mlist::Array{Array{T1, 2}} = [An, Gn, KL, Bn];
    Mlist[1] .= Xn / reduce(*, Mlist[2:end]);
    local Mhat = copy(Mlist);
    local nM = length(Mlist);
    local nLeft = 2; # number of matrix factors (Q in the paper)

    # pre-computed multilication and Lipschitz coeffs to speed-up
    local L::Array{Any} = ones(nM);
    local R::Array{Any} = ones(nM);
    local Lambda_max::Array{Float64} = ones(nM);

    Cx = (1 + τX);
    # solve Sylvester Equation with differential smoothness
    local Syl_A = (params.λL/Cx) .* Lap;
    local Syl_B = Diff * Diff';

    ## STEP 3: Reconstruction Task (Beginning of while loop):
    while (loss > params.threshold && noIter <= noIteration) || noIter <= 300
        noIter += 1;
                
        ## STEP 4: Step-size Initialization:
        γ = γ * (1 - ζ*γ);

        ## STEP 5: Pre-compute matrix multilication and Lipschitz coefficient
        for q = 1:nM
            if q > 1
                L[q] = L[q-1] * Mlist[q-1];
            end
            if q < nM
                R[nM-q] = Mlist[nM-q+1] * R[nM-q+1];
            end
            Lambda_max[q] = opnorm(L[q]' * L[q]) * opnorm(R[q] * R[q]');
        end

        ## STEP 7: Obtain Xhat:
        mul!(temp, Mlist[1], R[1]);

        # Cx*X + (λL) .* Lap * X * (DD') = AKB + τX*Xn
        Syl_C = (1/Cx) * (temp + τX .* Xn);
        if params.λL > 0
            Xhat = sylvd(Syl_A, Syl_B, Syl_C);
        else # if no regularizers
            Xhat = Syl_C;
        end

        # Solve with strict constraint S(X)=S(Y)
        Yhat = Mc .* Xhat + Y;                                  # Yhat = Y + Sc(Xhat)
        Xhat = real(Yhat);                                        # Xhat = Y + Sc(Xhat)

        ## STEP 8: Obtain matrix factors A_1...A_Q and B
        for i = 1:nM
            if i == nLeft + 1 # skip the Kernel matrix
                continue
            end
            # heuristics: update A_2 every $freq iterations for speedup and stability
            if (i == nLeft) && (noIter % freq) != 0
                continue
            end

            solveClosed!(i, nLeft, params, Mhat, Mlist, L, R, Lambda_max, Xn, γ);

        end

        # STEP 9: succesive convex approximation for Z and X
        axpby!(γ, Xhat, (1-γ), Xn);                         # Xn+1 = γXhat + (1-γ)Xn

        ## Termination Criteria Parameters:
        Xhat_new = Xn;
        loss = norm(Xhat_new - Xhat_prev)/norm(Xhat_prev);
        Xhat_prev = copy(Xhat_new);
        Mlist = copy(Mhat);
        if flag == 0
            push!(imgLoss, norm(img[Misses] - Xn[Misses])/norm(img[Misses]));
            if imgLoss[end] > 10
                println("Out of bound");
                return Mlist, Xn, imgLoss;
            end
            println("Main Iter ", noIter, ": Loss ", loss, "; NRMSE ", imgLoss[end], "; MAE ", maxTemp*mean(abs.(img[Misses] - Xhat_new[Misses])));
        else
            println("Main Task Iteration Number ", noIter, ": Loss Value ", loss, ".");
        end
    end
    println("Main Task Terminated at Iteration ", noIter, ": Loss Value ", loss, " for NRMSE ", imgLoss[end], ".");
    return Mlist, Xn, imgLoss;
end

function solveClosed!(
    i::Int64,
    nLeft::Int64,
    params::OptimizerParams,
    Mhat::Array{Array{T, 2}},
    Mlist::Array{Array{T, 2}},
    L::Array{Any},
    R::Array{Any},
    Lambda_max::Array{Float64},
    X::Array{T, 2},
    γ::Float64
    ) where {T <: Union{ComplexF64, Float64}}
    if i == nLeft + 1
        return
    end
    
    local Mn = copy(Mlist[i])

    if i <= nLeft
        solveforLeftClosed!(Mn, γ, params, X, L[i], R[i], i, length(Mlist));
        # solveforRightClosed!(Mn, γ, params, X, L[i], R[i], Lambda_max, i);
    else
        solveforLeftClosed!(Mn, γ, params, X, L[i], R[i], i, length(Mlist));
        # solveforRightClosed!(Mn, γ, params, X, L[i], R[i], Lambda_max, i);
    end

    Mhat[i] .= Mn;
end

function solveforLeftClosed!(
    Un::Array{T, 2},
    γ::Float64, 
    params::OptimizerParams, 
    X::Array{T, 2}, 
    L::Union{Array{T, 2}, Float64}, 
    R::Union{Array{T, 2}, Float64}, 
    i::Int,
    nM
    ) where {T <: Union{ComplexF64, Float64}}
    # H0 = copy(Un);
    # H1 = similar(H0);
    Lh = L';
    Rh = R';
    LhL = Lh*L;
    RRh = R*Rh;
    LhXRh = Lh*(X)*Rh;
    τ = params.τ[i];

    # solve Sylvester Equation LhLxAxRRh - LhXRh + alpha A + tau(A-Un) = 0
    # (tau+alpha) A + LhLxAxRRh = LhXRh + tau Un

    if (i == 1) || (i == nM)
        H1 = sylvd(1/(params.alpha[i]+τ) .* LhL, RRh, 1/(params.alpha[i]+τ) .* (τ*Un + LhXRh));
    else
        r, c = size(Un);

        local Mc_Un = sparse(BlockDiagonal(repeat([ones(Int(r/params.noK), Int(c/params.noK))], params.noK)));
        local dMc_Un = sparse(diagm(vec(Mc_Un)));

        local kLR = kron(transpose(RRh), LhL);
        local kI = Matrix{ComplexF64}(I, size(kLR)) .* (params.alpha[i]+τ);

        local RHS = dMc_Un * (vec(LhXRh) + τ*vec(Un));

        local dH1 = pinv(dMc_Un*kLR + kI) * RHS;

        H1 = reshape(dH1, size(Un));

    end

    axpby!(γ, H1, (1-γ), Un);
end

function solveforRightClosed!(
    Bn::Array{T, 2},
    γ::Float64,
    params::OptimizerParams,
    X::Array{T, 2},
    L::Union{Array{T, 2}, Float64},
    R::Union{Array{T, 2}, Float64},
    Lambda_max::Array{Float64},
    i::Int
    ) where {T <: Union{ComplexF64, Float64}}
    ## Hyperparameter Initialization:
    lossB = 1;
    noIterB = 0;
    local Lh = L';
    local Rh = R';
    local LhL = Lh*L;
    local RRh = R*Rh;
    local LhXRh = Lh*X*Rh;
    τ = params.τ[i];
    Lips = Lambda_max[i] + τ;
    λ = 1.98 * (1 - params.α)/Lips;                                          # λ = 0.99* 2[1-α]/Lb
    λλ1 = λ*params.λ1[i];                                                      # λλ1 Product
    α_1 = 1-params.α;

    r = size(Bn, 1);
    r = Integer(r / params.noK);

    ## Matrix Initialization:
    local TH0 = similar(Bn);
    local TH1 = similar(Bn);

    ## STEP 5: Initialization:
    local H0 = copy(Bn);
    local H1 = similar(H0);

    ## STEP 6: Computing T(B) and Tα(B)
    # pT = I_{Nker, Nker} - (1/Nker) * 1_{Nker, Nker}
    # aT = (1/Nker) * 1_{Nker, Nfr}
    # T(H0) = (I_{Nker, Nker} - (1/Nker) * 1_{Nker, Nker}) * H0 + (1/Nker) * 1_{Nker, Nfr}
    for i = 1:params.noK
        xStart = (i-1)*r + 1;
        xEnd = i*r;
        # TH0[xStart:xEnd, :] = pT * H0[xStart:xEnd, :] + aT; 
        TH0[xStart:xEnd, :] = affineMappingRight(H0[xStart:xEnd, :]);
    end
    axpby!(α_1, H0, params.α, TH0);                            # Tα(H0) = αT(H0) + (1- α)H0   
    
    ## STEP 7: H_(1/2) Update
    local ∇ = LhL*H0*RRh - LhXRh + τ * (H0 - Bn);
    local gradH0 = λ .* ∇;                                             # λ∇g(H0) = λ(K'Dn'DnKH0 + τB*H0 - K'Dn'X - τB*Bn)
    local H1_2 = TH0 - gradH0;                                       # H1_2 = T(H0) - λ∇g(H0)

    ## STEP 8: H_1 Update
    softThresholdingProximal!(H1, H1_2, λλ1);                   # [H1]_ij = [H1_2]_ij (1 - λλ1/max(|[H1_2]_ij|, λλ1))

    ## STEP 9: While Loop
    while (lossB > params.thresholdInner && noIterB < params.noIterationInner)
        ## STEP 10: H_(k+1/2) update
        # T(H1) = (I_{Nker, Nker} - (1/Nker) * 1_{Nker, Nker}) * H1 + (1/Nker) * 1_{Nker, Nfr}
        for i = 1:params.noK
            xStart = (i-1)*r + 1;
            xEnd = i*r;
            # TH1[xStart:xEnd, :] = pT * H1[xStart:xEnd, :] + aT; 
            TH1[xStart:xEnd, :] = affineMappingRight(H1[xStart:xEnd, :]);
        end
        # ∇g(H1) = λ [K'Dn'DnKH1 + τB*H1 - K'Dn'X - τB*Bn]
        ∇ = LhL*H1*RRh - LhXRh + τ * (H1 - Bn);
        local gradH1 = λ .* ∇;
        # H1_2 = H1_2 + T(H1) - ∇g(H1) - Tα(H0) + ∇g(H0)
        H1_2 += TH1 - gradH1 - TH0 + gradH0;
        TH0 = copy(TH1);
        H0 = copy(H1);
        gradH0 = copy(gradH1);
        axpby!(α_1, H0, params.α, TH0);

        ## STEP 11: H_(k+2) update: [H2]_ij = [H1_2]_ij (1 - λλ1/max(|[H1_2]_ij|, λλ1))
        softThresholdingProximal!(H1, H1_2, λλ1);

        ## Termination Criteria Update:
        noIterB += 1;
        lossB = norm(H1 - H0, 2)/norm(H0, 2);
    end
    #println("Solve for B Terminated at Iteration ", noIterB, " for Loss Value ", lossB, ".");
    axpby!(γ, H1, (1-γ), Bn);                         # Xn+1 = γXhat + (1-γ)Xn
end
