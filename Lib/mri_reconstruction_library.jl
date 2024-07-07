################################################################################################
## Reconstruction Tasks:
################################################################################################
## Library Dependencies:
using SparseArrays, MatrixEquations
using Random, LinearAlgebra, FFTW
include("./mri_support_functions_library.jl")

# Recovery Task Parameters:
mutable struct OptimizerParams
    Np :: Int64                         # Phase Encoding Lines
    Nf :: Int64                         # Phase Frequency Lines
    Nfr :: Int64                        # Number of Frames
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
    # C :: Array{Float64}                 # Bounding Constant parameter on Matrix A
    alpha :: Array{Float64}                 # Bounding Constant parameter on Matrix A
    # CA :: Float64                       # Bounding Constant parameter on Matrix A
    # CG :: Float64                       # Bounding Constant parameter on Matrix G
    τ :: Array{Float64}                 # Sub Optimisation Tasks parameter in general
    # τA :: Float64                       # Sub Optimization Tasks Parameter for || A - An ||
    # τG :: Float64                       # Sub Optimization Tasks Parameter for || G - Gn ||
    # τB :: Float64                       # Sub Optimization Tasks Parameter for || B - Bn ||
    τS :: Float64                       # Sub Optimization Tasks Parameter for || B - Bn ||
    α :: Float64                        # Step Size Parameter for Sub Optimization Tasks in general
    # αA :: Float64                       # Step Size Parameter for Sub Optimization Tasks
    # αG :: Float64                       # Step Size Parameter for Sub Optimization Tasks
    # αB :: Float64                       # Step Size Parameter for Sub Optimization Tasks
    αS :: Float64                       # Step Size Parameter for Sub Optimization Tasks
    noK :: Int64                        # Number of kernels employed
    threshold :: Float64                # Threshold of loss criterion for the Main Reconstruction Task                
    noIteration :: Int64                # Itertion Count Cut off in convergence criteria is 
                                        #    not met for the Main Reconstruction Task
    thresholdInner :: Float64           # Threshold of loss criterion for the Sub Optimization Tasks
    noIterationInner :: Int64           # Itertion Count Cut off in convergence criteria is 
                                        #    not met for the Sub Optimization Tasks
end

function MultiLKRIM(Y::Array{T1, 2}, KL, M::BitArray{2}, params::OptimizerParams, ImageData::Array{T2}) where {T1, T2 <:Union{Float64, ComplexF64}}
    ## Matrix Size initialization
    rng = MersenneTwister(seed);
    Np = params.Np;
    Nf = params.Nf;
    Nfr = params.Nfr;
    Nk = Np*Nf;

    (Nl1, Nl2) = size(KL);

    if size(ImageData, 1) != 1
        img = reshape(ImageData, Nk, Nfr);
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
    if params.λ2 > 0
        λz = params.λ3/params.λ2;
    else
        λz = 0;
    end

    ## Parameter, Matrix Values to save computation in time in loops:
    Mc = .!M;

    #Fourier Transform Plan
    pf  = plan_fft(zeros(Np, Nf, Nfr), [1 2], flags=FFTW.MEASURE);
    pft = plan_fft(zeros(Nk, Nfr), 2, flags=FFTW.MEASURE);
    
    ## STEP 1: Low-resolution FFT Reconstruction:
    Xlow = ifft2(Y, pf);

    ## STEP 2: Random Initialiation:
    # X0:
    Xn = copy(Xlow);
    Xhat = copy(Xlow);
    # Left of kernel matrix
    An = 1e-1*randn(rng, ComplexF64, Nk, d*params.noK);
    Dn = randn(rng, ComplexF64, 2, 4);
    Dn = sparse(BlockDiagonal(repeat([Dn], params.noK))); # (2*params.noK, d*params.noK)
    En = randn(rng, ComplexF64, 3, d);
    En = sparse(BlockDiagonal(repeat([En], params.noK))); # (2*params.noK, d*params.noK)
    Gn = randn(rng, ComplexF64, d, Int(Nl1/params.noK));
    Gn = sparse(BlockDiagonal(repeat([Gn], params.noK))); # (d*params.noK, Nl1)

    # Right of kernel matrix
    Bn = 1e-1*randn(rng, ComplexF64, Nl2, Nfr);
    # Auxiliary variable to calculate Xn more easily
    Zn = similar(Xlow);
    mul!(Zn, pft, Xn);

    # Others:
    Xhat_prev = copy(Xn);
    Xhat_new = similar(Xn);
    Zhat = similar(Zn);
    Aux = similar(Xn);
    
    local Mlist::Array{Array{T1, 2}} = [An, Gn, KL, Bn];
    Mlist[1] .= Xn * pinv(reduce(*, Mlist[2:end])); # initialize so that factorization is same as observed data
    local Mhat = copy(Mlist);
    local nM = length(Mlist);
    local nLeft = 2; # number of matrix factors on the left of kernel
    local kindices = [3]; # index of the kernel matrix

    # pre-computed multilication and Lipschitz coeffs to speed-up
    local L::Array{Any} = ones(nM);
    local R::Array{Any} = ones(nM);
    local Lambda_max::Array{Float64} = ones(nM);

    ## STEP 3: Reconstruction Task (Beginning of while loop):
    while (loss > params.threshold && noIter <= noIteration) || noIter <= 10
        noIter += 1;

        ## STEP 4: Initialization:
        γ = γ * (1 - ζ*γ);

        ## STEP 5: Pre-compute matrix multilication and Lipschitz coefficient
        for q = 1:nM
            if q > 1
                L[q] = L[q-1] * Mlist[q-1];
            end
            if q < nM
                R[nM-q] = Mlist[nM-q+1] * R[nM-q+1];
            end
            Lambda_max[q] = opnorm(L[q]' * L[q]) * opnorm(R[q] * R[q]'); # Lipschitz coeff
        end

        ## STEP 6: Obtain Zhat:
        if λz > 0 # if there is regularization on Ft(X)
            mul!(Aux, pft, Xn);                                 # Aux = Ft(Xn)
            axpby!(τZ/(1+τZ), Zn, 1/(1+τZ), Aux);
            softThresholdingProximal!(Zhat, Aux, λz);           # Zhat = Shrinkage operatore on Ft(X) attributed l1-norm on Zn
            ldiv!(Aux, pft, Zn);                                # Aux = invFt(Z)
        end

        ## STEP 7: Obtain Xhat:
        Cz = params.Nfr * params.λ2;
        Cx = 1/(1 + Cz + τX);
        
        mul!(Xhat, Mlist[1], R[1]);                         # Xhat = AGKB
        axpby!(Cx, Cz .* Aux + τX .* Xn, Cx, Xhat);         # Xhat = AGKB + λ2Nfr * invFt(Z) + τX .* Xn
        Yhat = fft2(Xhat, pf);                              # Yhat = F(Xhat)
        Yhat = Mc .* Yhat + Y;                                  # Yhat = Y + Sc(F(Xhat))
        Xhat = ifft2(Yhat, pf);                             # Xhat = invF(Y + Sc(F(Xhat)))

        ## STEP 8: Obtain matrix factors A_1...A_Q and B
        freq = 30;
        for i = 1:nM
            if i in kindices # skip the Kernel matrix
                continue
            end
            # heuristics: update every freq iterations to speedup and for stability
            if ((i+1) in kindices) && noIter%freq!=0
                continue
            end
            
            solveClosed!(i, kindices, nLeft, params, Mhat, Mlist, L, R, Lambda_max, Xn, γ);
        end

        # STEP 9: SCA for Z and X
        axpby!(γ, Zhat, (1-γ), Zn);                         # Zn+1 = γZhat + (1-γ)Zn
        axpby!(γ, Xhat, (1-γ), Xn);                         # Xn+1 = γXhat + (1-γ)Xn

        ## Termination Criteria Parameters:
        Xhat_new = Xn;
        loss = norm(Xhat_new - Xhat_prev)/norm(Xhat_prev);
        Xhat_prev = copy(Xhat_new);
        Mlist = copy(Mhat);
        if flag == 0
            push!(imgLoss, norm(img - Xn)/norm(img));
            println("Main Iter ", noIter, ": Loss ", loss, "; NRMSE ", imgLoss[end]);
            if imgLoss[end] > 10
                println("out of bound")
                return Mlist, Xn, imgLoss;
            end
        else
            println("Main Task Iteration Number ", noIter, ": Loss Value ", loss, ".");
        end
    end
    println("Main Task Terminated at Iteration ", noIter, " for NRMSE ", imgLoss[end], ".");
    return Mlist, Xn, imgLoss;
end

function solveClosed!(
    i::Int64,
    kindices,
    nLeft,
    params::OptimizerParams,
    Mhat::Array{Array{T, 2}},
    Mlist::Array{Array{T, 2}},
    L::Array{Any},
    R::Array{Any},
    Lambda_max::Array{Float64},
    X::Array{T, 2},
    γ::Float64,
    ) where {T <: Union{ComplexF64, Float64}}
    
    Mn = copy(Mlist[i]);

    if i <=nLeft
        solveforLeftClosed!(Mn, γ, params, X, L[i], R[i], i, length(Mlist));
    else
        solveforLeftClosed!(Mn, γ, params, X, L[i], R[i], i, length(Mlist));
        # solveforRightClosed!(Mn, γ, params, X, L[i], R[i], Lambda_max, i, kindices, length(Mlist));
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
    Lh = L';
    Rh = R';
    LhL = Lh*L;
    RRh = R*Rh;
    LhXRh = Lh*(X)*Rh;
    τ = params.τ[i];

    # solve Sylvester Equation LhLxAxRRh - LhXRh + alpha A + tau(A-Un) = 0
    # (tau+alpha) A + LhLxAxRRh = LhXRh + tau Un
    # print(size(LhL), size(RRh), size(Un), size(LhXRh), size(Sum))

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
    j::Int,
    kindices,
    nM
    ) where {T <: Union{ComplexF64, Float64}}
    ## Hyperparameter Initialization:
    lossB = 1;
    noIterB = 0;
    Lh = L';
    Rh = R';
    LhL = Lh*L;
    RRh = R*Rh;
    LhXRh = Lh*X*Rh;
    τ = params.τ[j];
    Lips = Lambda_max[j] + τ;
    λ = 1.98 * (1 - params.α)/Lips;                                          # λ = 0.99* 2[1-α]/Lb
    λλ1 = λ*params.λ1[j];                                                      # λλ1 Product
    α_1 = 1-params.α;

    (r, c) = size(Bn);
    r = Int(r / params.noK);
    if j<nM
        c = Int(c / params.noK);
    end

    if (j+1) in kindices
        affineMapping = affineMappingLeft;
    else
        affineMapping = affineMappingRight;
    end

    ## Matrix Initialization:
    TH0 = similar(Bn);
    TH1 = similar(Bn);

    ## STEP 5: Initialization:
    H0 = copy(Bn);
    H1 = similar(H0);

    ## STEP 6: Computing T(B) and Tα(B)
    # pT = I_{Nker, Nker} - (1/Nker) * 1_{Nker, Nker}
    # aT = (1/Nker) * 1_{Nker, Nfr}
    # T(H0) = (I_{Nker, Nker} - (1/Nker) * 1_{Nker, Nker}) * H0 + (1/Nker) * 1_{Nker, Nfr}
    for i = 1:params.noK
        xStart = (i-1)*r + 1;
        xEnd = i*r;
        yStart = (i-1)*c + 1;
        yEnd = i*c;
        # TH0[xStart:xEnd, :] = pT * H0[xStart:xEnd, :] + aT; 
        if j<nM
        TH0[xStart:xEnd, yStart:yEnd] = affineMapping(H0[xStart:xEnd, yStart:yEnd]);
        else
            TH0[xStart:xEnd, :] = affineMapping(H0[xStart:xEnd, :]);
        end
    end
    axpby!(α_1, H0, params.α, TH0);                            # Tα(H0) = αT(H0) + (1- α)H0   
    
    ## STEP 7: H_(1/2) Update
    ∇ = LhL*H0*RRh - LhXRh + τ * (H0 - Bn);
    gradH0 = λ .* ∇;                                             # λ∇g(H0) = λ(K'Dn'DnKH0 + τB*H0 - K'Dn'X - τB*Bn)
    H1_2 = TH0 - gradH0;                                       # H1_2 = T(H0) - λ∇g(H0)

    ## STEP 8: H_1 Update
    softThresholdingProximal!(H1, H1_2, λλ1);                   # [H1]_ij = [H1_2]_ij (1 - λλ1/max(|[H1_2]_ij|, λλ1))

    ## STEP 9: While Loop
    while (lossB > params.thresholdInner && noIterB < params.noIterationInner)
        ## STEP 10: H_(k+1/2) update
        # T(H1) = (I_{Nker, Nker} - (1/Nker) * 1_{Nker, Nker}) * H1 + (1/Nker) * 1_{Nker, Nfr}
        for i = 1:params.noK
            xStart = (i-1)*r + 1;
            xEnd = i*r;
            yStart = (i-1)*c + 1;
            yEnd = i*c;
            # TH1[xStart:xEnd, :] = pT * H1[xStart:xEnd, :] + aT; 
            if j<nM
                TH1[xStart:xEnd, yStart:yEnd] = affineMapping(H1[xStart:xEnd, yStart:yEnd]);
            else
                TH1[xStart:xEnd, :] = affineMapping(H1[xStart:xEnd, :]);
            end
        end
        axpby!(α_1, H1, params.α, TH1);                 # Tα(H1) = αT(H1) + (1- α)H1

        # ∇g(H1) = λ [K'Dn'DnKH1 + τB*H1 - K'Dn'X - τB*Bn]
        ∇ = LhL*H1*RRh - LhXRh + τ * (H1 - Bn);
        gradH1 = λ .* ∇;
        # H1_2 = H1_2 + T(H1) - ∇g(H1) - Tα(H0) + ∇g(H0)
        H1_2 += TH1 - gradH1 - TH0 + gradH0;


        ## STEP 11: H_(k+2) update: [H2]_ij = [H1_2]_ij (1 - λλ1/max(|[H1_2]_ij|, λλ1))
        softThresholdingProximal!(H1, H1_2, λλ1);

        TH0 = copy(TH1);
        H0 = copy(H1);
        gradH0 = copy(gradH1);

        ## Termination Criteria Update:
        noIterB += 1;
        # lossB = norm(H1 - H0, 2)/norm(H0, 2);
    end
    # println("Solve for B Terminated at Iteration ", noIterB, " for Loss Value ", lossB, ".");

    axpby!(γ, H1, (1-γ), Bn);                         # Xn+1 = γXhat + (1-γ)Xn
end
