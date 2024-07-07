################################################################################################
#= Step 0: Including code library of functions required for
        execution. Landmark, Kernel, RSE and Reconstruction
        based library.
=#

# import Pkg;
# Pkg.add("MAT")
# Pkg.add("LinearAlgebra")
# Pkg.add("BlockDiagonals")
# Pkg.add("Statistics")
# Pkg.add("Dates")
# Pkg.add("Distances")
# Pkg.add(name="ToeplitzMatrices", version="0.7.0")
# Pkg.add("Clustering")
# Pkg.add("MatrixEquations")
# Pkg.add("FFTW")
# Pkg.add("StatsBase")

using Dates, MAT, LinearAlgebra, BlockDiagonals, Statistics;
include("./Lib/landmark_library.jl")
include("./Lib/kernel_library.jl")
include("./Lib/dimension_reduction_library.jl")
include("./Lib/mri_analysis_library.jl")
include("./Lib/mri_reconstruction_library.jl")

server_name = "personal";
dataname = "MRXCAT-408408360.mat";

usamp_type = "retrospective";

masks = [
    "Mask408408360-8xC.mat",
];
println(now());

# test each mask
for m = masks
    global maskname=m;
    global seed = 2007;
    include("MultiLKRIM.jl");
end