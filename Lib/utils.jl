using LinearAlgebra;
include("kernel_library.jl");

function diff_operator(T::Int64, type="diff")
    # Temporal differential operator Diff
    # ToeplitzMatrices indexing works for v0.7.0 only
    v = zeros(Float64, T);
    v[1] = -1.0;
    v[2] = 1.0;
    Diff = TriangularToeplitz(v, :L);
    Diff = Diff[:, 1:end-1];

    if type == "diff"
        return Diff;
    end
    return Matrix{Float64}(I, T, T);
end

function get_kNN_pos(Pos::Array{dtype, 2}, k::Int64, alpha=0.0, beta=1.0) where {dtype <:Union{Float64, ComplexF64}}
    N = size(Pos, 1);
    Dist = zeros(N, N);
    for i = 1:N
        for j = i+1:N
            Dist[i, j] = norm(Pos[i, :] - Pos[j, :]);
            Dist[i, j] = Dist[i, j] * (0.5+abs(Pos[i, 1] - Pos[j, 1]) / (abs(Pos[i, 1] - Pos[j, 1]) + abs(Pos[i, 2] - Pos[j, 2])));
            Dist[j, i] = Dist[i, j];
        end
    end

    W = zeros(N, N);
    KNN = zeros(Int64, N, k);
    for i = 1:N
        ind = partialsortperm(Dist[i, :], 2:k+1); # get k-nn
        KNN[i, :] .= ind;
        W[i, ind] .= 1 ./ ((Dist[i, ind]).^2);
        W[ind, i] .= W[i, ind];
    end
    ldiv!(maximum(W), W);

    Deg = diagm(W * ones(Float64, N));               # Degree matrix [D]_nn = ∑K_nn'
    Lap = Deg - W;                                     # unnormalized Laplacian

    if beta != 0
        Lap = (Lap + alpha.*Matrix{Float64}(I, N, N))^beta;
    else
        Lap = (Lap + alpha.*Matrix{Float64}(I, N, N));
    end

    return KNN, W, Lap, Dist;
end

function get_kNN_kernel(K::Array{dtype, 2}, k::Int64) where {dtype <:Union{Float64, ComplexF64}}
    (N, N) = size(K); # num nodes x num features
    Dist = zeros(N, N);
    for i = 1:N
        for j = i+1:N
            Dist[i, j] = sqrt(abs(K[i, i] - 2*K[i,j] + K[j, j]));
            Dist[j, i] = Dist[i, j];
        end
    end

    W = zeros(N, N);
    KNN = zeros(Int64, N, k);
    sigma = mean(Dist);
    for i = 1:N
        ind = partialsortperm(Dist[i, :], 2:k+1); # get k-nn
        KNN[i, :] .= ind;
        # W[i, ind] .= 1 ./ ((Dist[i, ind]).^2);
        W[i, ind] .= exp.(-(Dist[i, ind].^2) / (sigma^2));
        W[ind, i] .= W[i, ind];
    end
    ldiv!(maximum(W), W);

    # W = diagm(1 ./sum(W, dims=2)[:, 1]) * W;

    Deg = diagm(W * ones(Float64, N));               # Degree matrix [D]_nn = ∑K_nn'
    Lap = Deg - W;                                     # unnormalized Laplacian

    if beta != 0
        Lap = (Lap + alpha.*Matrix{Float64}(I, N, N))^beta;
    else
        Lap = (Lap + alpha.*Matrix{Float64}(I, N, N));
    end

    return KNN, W, Lap, Dist;
end

function left_pad(X::Array{dtype, 2}, d::Int64) where {dtype <:Union{Float64, ComplexF64}}
    #=
    Pad d zeros columns on the left
    =#
    (N, T) = size(X);
    X_padded::Array{dtype, 2} = zeros(dtype, N, T+d);
    X_padded[:, d+1:T+d] .= X;
    return X_padded;
end

function right_pad(X::Array{dtype, 2}, d::Int64) where {dtype <:Union{Float64, ComplexF64}}
    #=
    Pad d zeros columns on the right
    =#
    (N, T) = size(X);
    X_padded::Array{dtype, 2} = zeros(dtype, N, T+d);
    X_padded[:, 1:T] .= X;
    return X_padded;
end

function pad(X::Array{dtype, 2}, d::Int64) where {dtype <:Union{Float64, ComplexF64}}
    #=
    Pad d zeros rows
    =#
    (N, T) = size(X);
    X_padded::Array{dtype, 2} = zeros(dtype, N+d, T);
    X_padded[:N, :] .= X;
    return X_padded;
end

function get_window(X, dt)
    (N, T) = size(X);
    X_padded = left_pad(X, dt);
    X_padded = right_pad(X_padded, dt);
    Windows = zeros(N*(2*dt+1), T);
    for t = dt+1:T+dt
        t0 = t - dt;
        t1 = t + dt;
        Windows[:, t-dt] .= vec(X_padded[:, t0:t1]);
    end

    return Windows;
end

function get_patch(X::Array{dtype, 2}, KNN::Array{Int64, 2}, k::Int64, dt::Int64) where {dtype <:Union{Float64, ComplexF64}}
    #=
    Generate patches
    X: data matrix NxT
    KNN: k-nn matrix Nxk
    k: number of nearest neighbors
    dt: time window
    =#
    println("Start building patches.");

    (N, T) = size(X);
    @assert size(KNN, 1) == N;
    @assert size(KNN, 2) == k;

    X_padded = left_pad(X, dt);
    X_padded = right_pad(X_padded, dt);
    Patch::Array{dtype, 3} = zeros(N, T, k*(2*dt+1));

    for n = 1:N
        for t = dt+1:T+dt
            t0 = t - dt;
            t1 = t + dt;
            Patch[n, t-dt, :] .= vec(X_padded[KNN[n, :], t0:t1]);
        end
    end

    # A Toeplitz matrix
    # Mtime = falses(T, T);
    # for t = 1:T
    #     p = max(1, t-dt):min(t+dt, T);
    #     Mtime[t, p] .= 1;
    # end

    return Patch;
end

function get_nav(Y, Y_T, nav_type, sampling_pattern)
    """
    under construction
    """
    if occursin("SLP", dataname)
        if sampling_pattern == "every1Snapshot"
            if nav_type == "node"
                org_Ynav = copy(Y_T);
            else
                org_Ynav = copy(Y[Y.>0]);
                org_Ynav = reshape(org_Ynav, :, T);
            end
        else
            if nav_type == "node"
                org_Ynav = copy(Y_T[SnapshotSelect, :]);
            else
                org_Ynav = copy(Y[:, SnapshotSelect]);
            end
        end
    elseif nav_type == "patches"
        Patches = get_patch(Y[:, 1:50], KNN, k, dt); # N x T x patch_len
        org_Ynav = copy(reshape(Patches, N*50, :)'); # patch_len x (NT), (NT) is num of landmark points
    elseif nav_type == "window"
        Windows = get_window(Y, dt);
        org_Ynav = copy(Windows);
    elseif nav_type == "time"
        org_Ynav = copy(Y);
    end
end

function Normalize!(X)
    Y = X ./ sqrt.(sum(abs2, X, dims=1));
    X .= Y;
end