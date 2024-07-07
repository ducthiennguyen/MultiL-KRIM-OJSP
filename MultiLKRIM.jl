# ###############################################################################################
# STEP 0: Global Variales:
# (Describing the type of data, sampling trajectory in use)
@__DIR__
server_name = server_name;
dataname = dataname;

println("Server name: ", server_name);
println("Data name: ", dataname);

################################################################################################
#= Step 0: Working, Data, Output Directory Initialization & MAT file extrseraction: 
    (requires the file 'server_specific_data_access.m')
    The following code should make the undersampled 
    (retrospective or prospective) 
    k-space data {Y}, navigator (centra kspace) data {Ynav} with dimensions 
    Np, Nf, Nfr, Nnav and the mask trajectory (Mask) available.
    
    Outputs Available: Y, Ynav, Mask, Np, Nf, Nfr, Nnav, 
                       ImageData (if retrospective undersampling)
=#
include("server_specific_data_access.jl");
println("Data Initialization Complete ...");

################################################################################################
# Step 0: Parameter Initialization:
include("parameter_setup_PRO_0.jl");
println("Step 0: Parameter Initialization Complete ..."); println();

include("Lib/utils.jl");

################################################################################################

## Task Commencement:

sampling_pattern = "every1Snapshot";
nav_type = "time"; 

# num of nearest neighbors
k = 5;

# frequency for inner loops
freq = 30;

# generate knn graph by geo-position
KNN, W, Lap, Dist = get_kNN_pos(Pos, k, eps);

# Temporal differential operator Diff
Diff = diff_operator(T, diffType);

# demo with sampling ratio = 0.1
for srate in [0.1]
    # sampling when there is invalid data
    Mask = falses(N, T);
    SampleMatrix0 = (Temperature .> 0);

    if sampling_pattern == "entireSnapshots"
        SampleNum = Int(ceil(T*srate));
        SnapshotSelect = sample(rng, 1:T, SampleNum, replace=false);
        Mask[:, SnapshotSelect] .= 1;
        Mask = Mask .* SampleMatrix0;
    elseif sampling_pattern == "every1Snapshot"
        for i = 1:T
            SampleNum = Int(ceil(N*srate));
            IndexMeasure = findall(SampleMatrix0[:, i]);
            IndexSelect = sample(rng, IndexMeasure, SampleNum, replace=false);
            Mask[IndexSelect, i] .= 1;
        end
        Mask = Mask .* SampleMatrix0;
    end

    Misses = BitArray(SampleMatrix0 - Mask);

    local Y = Mask .* TemperatureScaled;
    local Y_T = zeros(T, N);
    transpose!(Y_T, Y);
    dt = 20; # time window
        
    local org_Ynav = copy(Y);
    local org_Ynav = reshape(org_Ynav, :, T);

    local Ynav = [org_Ynav];

    local Λ = landmarkExtraction(Ynav, noLandmark, ltype);

    local K = MultiKernelConstruct(Λ, ktype);

    local KL = BlockDiagonal(K);
    println(size(KL));

    local start_time = now();
    @time global Mlist, Xn, imgLoss = MultiLKRIM(Y, KL, Lap, Diff, Mask, Misses, param, TemperatureScaled);
    local end_time = now()-start_time;

    local reconImageData = broadcast(*, Xn, maxTemp);
    local reconImageData1 = reshape(reconImageData, N, T);
    reconImageData = broadcast(*, reduce(*, Mlist), maxTemp);
    local reconImageData2 = reshape(reconImageData, N, T);
    reconImageData2 = Mask .* Temperature + Misses .* reconImageData2;
    local Mc = SampleMatrix0;
    err1 = norm(reconImageData1[Mc]-Temperature[Mc])/norm(Temperature[Mc]);
    err2 = norm(reconImageData2[Mc]-Temperature[Mc])/norm(Temperature[Mc]);
    mae1 = mean(abs.(reconImageData1[Mc]-Temperature[Mc]));
    mae2 = mean(abs.(reconImageData2[Mc]-Temperature[Mc]));
    rmse1 = sqrt(mean((reconImageData1[Mc]-Temperature[Mc]).^2));
    rmse2 = sqrt(mean((reconImageData2[Mc]-Temperature[Mc]).^2));
    rmse = min(rmse1, rmse2);
    mape1 = mean(abs.(reconImageData1[Mc]-Temperature[Mc]) ./ abs.(Temperature[Mc]));
    mape2= mean(abs.(reconImageData2[Mc]-Temperature[Mc]) ./ abs.(Temperature[Mc]));
    mape = min(mape1, mape2);

    println("$(mae1) $(rmse) $(mape)")

    out_path = "./output/$(sampling_pattern)/SLP_$(Int(floor(srate*100)))percent.mat";
    matwrite(out_path, Dict(
        "Mask" => Mask,
        "Xn" => Xn,
        "Mlist" => Mlist,
        "imgLoss" => imgLoss,
        "alpha" => alpha,
        "lambda1" => λ1,
        "lambdaL" => λL,
        "knn" => k,
        "MAE" => [mae1, mae2],
        "NRMSE" => [err1, err2],
        "tauX" => τX,
        "srate" => srate,
        "dim" => d,
        "ltype" => ltype,
        "noLandmark" => noLandmark,
        "kernel" => kernel,
        "diffType" => diffType,
        "eps" => eps,
        "run_time" => end_time.value/1000,
        "nav_type" => nav_type,
        "freq" => freq
    ));

    reconImageData = nothing;
    reconImageData1 = nothing;
    reconImageData2 = nothing;
    Mlist = nothing;
    Xn = nothing;
    imgLoss = nothing;
    GC.gc();
end

################################################################################################
# Display Code Attribtes:
println("Server Name: ", server_name)
println("Data Name: ", dataname)

println("LANDMARK SELECTION ATTRIBUTES:")
println("No. of landmark points: ", noLandmark);
println("Method of landmark selection: ", ltype); println();

println("KERNEL GENERATION ATTRIBUTES")
dump(ktype); println()

println("DIMENSION REDUCTION ATTRIBUTES")
dump(dparams); println()

println("RECONSTRUCTION ATTRIBUTES:")
dump(param); println()