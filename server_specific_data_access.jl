################################################################################################
## Library Dependencies
using MAT, SparseArrays, FFTW, LinearAlgebra, StatsBase

################################################################################################
## Working Directory Initialization based on the server name:
if server_name == "graph"
    # Working Directory Initialization and changing to the working directory: (TVGS)
    workdir = "./";
    cd(workdir);
    # Location of the data:
    datadir = workdir * "/";
    # Directory for outputs:
    scratch = workdir * "/";
    system = "personal";
end

###############################################################################################
## Initialization of the kspace data:
## MAT Variable Extracion :
reader = matopen(joinpath(datadir, dataname));
data = read(reader);
close(reader);

if dataname == "Sensor.mat"
    Temperature = data["Temperature"];
    Temperature[Temperature.<10] .= 0.0;
    Temperature = Temperature[[1:4; 6:end], 1001:1500];
    Pos = data["Position"];
    Pos = Pos[[1:4; 6:end], :];
elseif occursin("SLP", dataname)
    Temperature = data["x_matrix"];

    # recast to Float64
    Temperature = Float64.(Temperature);

    # get Position
    Pos = data["Position"];

    # choose a subset of data
    N, T = 500, 400;
    Temperature = Temperature[:, 1:T];
elseif occursin("Temperature", dataname)
    N, T = 100, 500;
    Temperature = data["Data"][1:N, 1:T];
    Temperature = max.(Temperature, 0);
    Pos = data["Position"][1:N, :];
else
    N, T = 93, 200;
    Temperature = data["myDataPM"][1:N, 1:T];
    Temperature = max.(Temperature, 0);
    Pos = data["Position"][1:N, :];
end

N, T = size(Temperature);
maxTemp = maximum(abs.(Temperature));
minTemp = minimum(abs.(Temperature[Temperature.>0]));
TemperatureScaled = broadcast(/, Temperature, maxTemp);

# Empty variables:
data = nothing;
reader = nothing;
###############################################################################################
