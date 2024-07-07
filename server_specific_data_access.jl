################################################################################################
## Library Dependencies
using MAT, SparseArrays, FFTW, LinearAlgebra, StatsBase

################################################################################################
## Working Directory Initialization based on the server name:
if server_name == "personal"
    # Working Directory Initialization and changing to the working directory: (PERSONAL)
    workdir = "/home/thien/Desktop/";
    cd(workdir);
    # Location of the data:
    datadir = workdir * "/KRIM/KRIM";
    # Directory for outputs:
    scratch = workdir * "/KRIM";
    system = "personal";

    outputdir = joinpath(scratch, "kernel_out/MultiL-KRIM-dMRI/") # save path
    if ~isdir(outputdir)
        mkdir(outputdir)
    end
end

################################################################################################
# MAT Variable Extracion (Mask matrix):
reader = matopen(joinpath(datadir, "Mask", maskname));
data = read(reader);
close(reader);
if issparse(data["Mask"])
    Mask = Matrix(data["Mask"]);
else
    Mask = data["Mask"];
end

###############################################################################################
## Initialization of the kspace data:
## MAT Variable Extracion :
reader = matopen(joinpath(datadir, "ImageData", dataname));
data = read(reader);
close(reader);
if usamp_type == "retrospective"
    # (ImageData (Image Domain data), Navigator Data):

    if occursin("CARDIAC", dataname)
        ImageData::Array{ComplexF64} = data["seq"];
    else
        ImageData::Array{ComplexF64} = data["ImageData"];
    end

    # Image Parameters initilization:
    if length(size(ImageData)) == 3 # single coil data
        ImageData = ImageData[:, :, 1:Nfr];
        (Np, Nf, Nfr) = size(ImageData);
        param.Np = Np;
        param.Nf = Nf;
        param.Nfr = Nfr;
        Nc = 1;
    else # multi coil data
        (Np, Nf, Nfr, Nc) = size(ImageData);
    end

    # Mask reshape: 
    if length(size(Mask)) == 3
        Mask = Mask[:, :, 1:Nfr];
        Mask = reshape(Mask, Np*Nf, Nfr);
    else
        Mask = Mask[:, 1:Nfr];
    end

    # Mask Type Cast:
    Mask = .!iszero.(Mask);

    # k-Space Retrospective Undersampling:
    ImageData = ImageData/maximum(abs.(ImageData));
    Y = fft(ImageData, [1 2]);
    temp = copy(Y);
    Y = reshape(Y, (Np*Nf, Nfr));
    Y = Mask.*Y;

    # form the navigator data
    org_Ynav = temp[[1, 2, Np-1, Np], :, 1:Nfr];
    org_Ynav = reshape(org_Ynav, :, Nfr);
    temp = nothing;

elseif usamp_type == "prospective"
    ## MAT Variable Extracion (ImageData (Image Domain data), Navigator Data):
    reader = matopen(joinpath(datadir, dataname));
    data = read(reader);
    close(reader);
    const Ynav = data["Ynav"];
    Y = data["Y"];

    # Image Parameters initilization:
    if length(size(Y)) == 3
        (Np, Nf, Nfr) = size(Y);
        Nc = 1;
    elseif length(size(Y)) == 4
        (Np, Nf, Nfr, Nc) = size(Y);
    end
    Nnav = trunc(Int, size(Ynav, 1)/Nf);
    # Mask reshape: 
    if length(size(Mask)) == 3
        Mask = reshape(Mask, Np*Nf, Nfr);
    end

    # Mask Type Cast:
    Mask = .!iszero.(Mask);

    for i = 1:Nc
        Y[:,:,:,i] = Y[:,:,:,i]/(maximum(abs.(Y[:,:,:,i]))*Np*Nf);
    end
    
    # k-space Vectorization: (As the kspace is acquired from scanner; It is already undrersampled)
    Y = reshape(Y, Np*Nf, Nfr, Nc);
    
end

# Empty variables:
data = nothing;
reader = nothing;
###############################################################################################
