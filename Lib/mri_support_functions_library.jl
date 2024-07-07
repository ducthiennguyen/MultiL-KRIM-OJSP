################################################################################################
## Miscallenous MRI Support Functions:
################################################################################################

## Library Dependencies:
using LinearAlgebra, FFTW

#= 2D Fourier Transform designed for the Casorati Matrix:
    Input: A = 2D matrix, Casorati form of the Dynamic Image time-series
       N1 = length dimesnion
       N2 = width dimension
Ouptut: Aux = 2D martix, Casorati form of the fourier transoform of matrix A.
=#
function fft2(A::Array{T, 2}, p::FFTW.cFFTWPlan) where {T<:Union{ComplexF64,Float64}}
    (N1, N2, N3) = size(p);
    A = reshape(A, N1, N2, N3);
    output = similar(A);
    mul!(output, p, A);
    output = reshape(output, N1*N2, N3);
    return output;
end

#= 2D Inverse Fourier Transform designed for the Casorati Matrix:
Input: A = 2D matrix, Casorati form of the Dynamic kspace time-series
       N1 = length dimesnion
       N2 = width dimension
Ouptut: Aux = 2D martix, Casorati form of the inverse fourier transoform of matrix A.
=#
function ifft2(A::Array{T, 2}, p::FFTW.cFFTWPlan) where {T<:Union{ComplexF64,Float64}}
    (N1, N2, N3) = size(p);
    A = reshape(A, N1, N2, N3);
    output = similar(A);
    ldiv!(output, p, A);
    output = reshape(output, N1*N2, N3);
    return output;
end
