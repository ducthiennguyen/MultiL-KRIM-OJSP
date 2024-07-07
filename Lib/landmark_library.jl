################################################################################################
## Importing Libraries required for the tasks:
using Distances, Random, Clustering

################################################################################################
## Landmark Extraction Related Codes:
################################################################################################

function genYnav(Ynav::Array{T, 3}, w::Int) where {T<:Union{ComplexF64,Float64}}
    local (Nnav, Nf, Nfr) = size(Ynav);
    #Ynav_window = zeros(ComplexF64, Nnav*w*Nf, Nfr);
    Ynav_window = nothing;
    for i = 1:Nfr-w+1
        ynav = Ynav[:, :, i:i+w-1];
        ynav = reshape(ynav, Nnav*Nf*w, 1);
        if i == 1
            Ynav_window = ynav;
        else
            Ynav_window = hcat(Ynav_window, ynav);
        end
    end
    return Ynav_window;
end

function makeKL(K::Array{Array{T, 2}}) where {T<:Union{ComplexF64,Float64}}
    Nker = length(K);
    local data = zeros(ComplexF64, Nker*size(K, 1), Nker*size(K ,2));
    eye = Matrix{Float64}(1.0I, Nker, Nker);
    for i = 1:Nker
        if i == 1
            data = kron(eye[i, :], K[:, :, i]);
        else
            data = hcat(data, kron(eye[i, :], K[:, :, i]));
        end
    end
    return data;
end

#= LANDMARK POINT SELECTION USING MAXMIN/RANDOM ALGORITHM:
Detect landmark points using either a random  or Maxmin algorithm:
Inputs: data (real or complex matrix) - Data matrix with columns as the data points 
        noLandmark (int) - The number of landmark points to be identified from the data matrix
        type (string) - A String either "random" or "maxmin" determining the algorithm to use
Output: lambda (same as data) - Contains columns of data identified as the landmark points
=#
function landmarkExtraction(data::Array{Array{T, 2}}, noLandmark::Array{Int}, type::String = "maxmin") where {T<:Union{ComplexF64,Float64}}
    landmark::Array{Array{T, 2}} = [];
    if type == "maxmin"
        for (i, ynav) in enumerate(data)
            ldiv!(maximum(abs.(ynav)), ynav);
            lm = maxminLandmark(ynav, noLandmark[i]);
            ldiv!(maximum(abs.(lm)), lm);
            push!(landmark, lm);
        end
    elseif type == "random"
        for (i, ynav) in enumerate(data)
            N = size(ynav, 2);
            lidx = rand(1:N, noLandmark[i]);
            lm = ynav[:, lidx];
            ldiv!(maximum(abs.(lm)), lm);    
            push!(landmark, lm);
        end
    elseif type == "kmeans"
        for (i, ynav) in enumerate(data)
            ldiv!(maximum(abs.(ynav)), ynav);
            local R = kmeans(ynav, noLandmark[i]; maxiter=200)

            @assert nclusters(R) == noLandmark[i] # verify the number of clusters

            # a = assignments(R) # get the assignments of points to clusters
            # c = counts(R) # get the cluster sizes
            lm = R.centers # get the cluster centers
            ldiv!(maximum(abs.(lm)), lm);
            push!(landmark, lm);
        end
    elseif type == "fuzzy"
        for (i, ynav) in enumerate(data)
            ldiv!(maximum(abs.(ynav)), ynav);
            local R = fuzzy_cmeans(ynav, noLandmark[i], 1.05; maxiter=200)

            @assert nclusters(R) == noLandmark[i] # verify the number of clusters

            # a = assignments(R) # get the assignments of points to clusters
            # c = counts(R) # get the cluster sizes
            lm = R.centers # get the cluster centers
            matwrite("fuzzy_sea_temp.mat", Dict(
                "centers" => lm,
                "weights" => R.weights
            ))
            ldiv!(maximum(abs.(lm)), lm);
            push!(landmark, lm);
        end
    end
    return landmark;
end

#= LANDMARK POINT SELECTION USING MAXMIN ALGORITHM:
Detect landmark points using the Maxmin algorithm:
Inputs: data (complex matrix) - Data matrix with columns as the data points 
        noLandmark (int) - The number of landmark points to be identified from the data matrix
Output: lambda (same as data) - Contains columns of data identified as the landmark points
=#
function maxminLandmark(data::Array{T, 2}, noLandmark::Int64) where {T<:Union{ComplexF64,Float64}}
    #print("Welcome to maxminLandmark..")
    ## Number of data points (columns) in the data matrix:
    N = size(data, 2);
    ## Random inital points to start with:
    local seed = 2;
    ## Extracting seed points for the maxmin algorithm:
    lidx = rand(1:N, seed);
    lambda = data[:, lidx];
    ## Compute pairwise distances and :
    # local K = ExponentialKernelComplex(data, 0.8, "gaussian"); # Gaussian Kernel matrix constructed from navigator data
    # local K = PolynomialKernelComplex(data);

    # local d = zeros(Float64, N, N);
    # for i = 1:N
    #     for j = 1:N
    #         d[i, j] = sqrt(real(K[i, i] + K[j, j] - K[i, j] - K[j, i])); # RKHS distance between points of index i and j
    #         # d[i, j] = abs(K[i, j] / sqrt(abs.(K[i, i] * K[j, j])));
    #     end
    # end
    # distance = minimum(d[lidx, :], dims=1);
    distance = minimum(pairwise(Minkowski(2), lambda, data, dims=2), dims=1);
    for i = (seed+1):noLandmark
        ## select the landmark point farthest from the already selected group of landmark points:
        maxDistance = maximum(distance);
        # println(maxDistance)
        S = vec(distance .== maxDistance);
        idx = findfirst(S);
        ## add the newly identifed landmark point to the data list:
        lambda = hcat(lambda, data[:, idx]);
        # println(size(lambda))
        ## compute the distance of the selected landmark point from the remaining points:
        distance1 = pairwise(Minkowski(2), reshape(lambda[:, i], length(lambda[:, i]), 1), data, dims=2);
        # distance1 = reshape(d[i, :], 1, N);
        distance = min.(distance, distance1);
    end
    ## return the identified landmark data points:
    return lambda
end
################################################################################################
