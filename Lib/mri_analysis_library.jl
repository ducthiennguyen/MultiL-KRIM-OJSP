################################################################################################
## Miscallenous Functions:
################################################################################################
## Library Dependencies:
using LinearAlgebra

#= NRMSE Computation:
Input: OriginalImage = 3D Matrix, Image Domain Original Data (Ground Truth).
       ReconImage = 3D Matrix, Reconstructed Image (reconstruction from undersample kspace)
Output: error = Float numeric, Normalised Root Mean Square Error
=#
function checkError(OriginalImage::Array{T1, 3}, ReconImage::Array{T2, 3}) where {T1,T2<:Union{ComplexF64,Float64}}
    ldiv!(maximum(abs.(ReconImage)), ReconImage);
    N = size(OriginalImage, 3);
    E = OriginalImage - ReconImage;
    NormRef = [norm(OriginalImage[:, :, j], 2) for j = 1:N];
    NormDiff = [norm(E[:, :, j], 2) for j = 1:N];
    error = sum(NormDiff)/sum(NormRef);
end