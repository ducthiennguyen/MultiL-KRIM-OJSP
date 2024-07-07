# ###############################################################################################
# STEP 0: Global Variales:
# (Describing the type of data, sampling trajectory in use)
@__DIR__
server_name = "personal";
dataname = dataname;
maskname = maskname;
usamp_type = "retrospective";

k1 = findfirst(isequal('-'), dataname);
namedata = dataname[1:(k1-1)];
k1 = findfirst(isequal('-'), maskname);
k2 = findfirst(isequal('.'), maskname);
maskId = maskname[(k1+1):(k2-1)];

println("Server name: ", server_name);
println("Data name: ", dataname);
println("Mask Name: ", maskname);
println("Sampling Method: ", usamp_type);

################################################################################################
# Step 0: Parameter Initialization:
include("parameter_setup_PRO_0.jl");
println("Step 0: Parameter Initialization Complete ..."); println();

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

# set output filename not to overwrite
obsNo = 1;
while true
    global fname = string(system, "_", namedata,  "_", maskId, "_",  obsNo, ".mat");
    global filename = joinpath(outputdir, fname);
    if isfile(filename) # don't overwrite
        global obsNo = obsNo + 1;
    else
        break;
    end
end
println("Observation Number: ", obsNo);

################################################################################################

## Task Commencement:
for i = 1:Nc
    println("Task for Nc number: ", i, " commences ...");
    if Nc == 1
        Ynav::Array{Array{ComplexF64, 2}} = [];
        push!(Ynav, org_Ynav);
        global Λ = landmarkExtraction(Ynav, noLandmark, ltype);
    else
        global Λ = landmarkExtraction(Ynav[:,:,i], noLandmark, ltype);
    end
    println("Step 1: Landmark Extraction Complete ..."); println()

    ################################################################################################
    #= Step 2: Kernel Consrtuctions using the identified landmark points.
            Strategies: Single Kernels - Gaussian, Linear, Polynomial
                        Multi Kernels - Combo of the above mentioned Kernels
                        Kernels constructed for both real/complex sensor data
    =#
    global K = MultiKernelConstruct(Λ, ktype);
    println("Step 2: Kernel Constrution Complete ..."); println()

    global KL = BlockDiagonal(K);
    println(size(KL));
    println("Step 3: Constructing KL Completed ..."); println()

    ################################################################################################
    #= Step 4: Reconstruction framework to determine D. B using the 
            FMHSDM + FLEXA optimization frameworks:
    =#
    println("MRI Reconstrution Commencing ....");

    if usamp_type == "retrospective"
        @time global Mlist, Xn, imgLoss = MultiLKRIM(Y, KL, Mask, param, ImageData);
    elseif usamp_type == "prospective"
        if Nc != 1
            @time global X, D, B, errConv = MriReconMKBiLMDM(Y[:,:,i], Khat, Mask, param, [-1.0]);
        else
            @time global X, D, B, errConv = MriReconMKBiLMDM(Y[:,:,1], Khat, Mask, param, [-1.0]);
        end
    end
    println("Step 4: MRI Reconstruction Complete ..."); println()

    println("Tasks for Nc = ", i, " finished.");
end
################################################################################################
# Error Analysis and logging parameters:
if usamp_type == "retrospective"
    reconImageData = Xn;
    reconImageData1 = reshape(reconImageData, Np, Nf, Nfr);
    reconImageData = reduce(*, Mlist);
    reconImageData2 = reshape(reconImageData, Np, Nf, Nfr);
    err1 = norm(reconImageData1-ImageData)/norm(ImageData);
    err2 = norm(reconImageData2-ImageData)/norm(ImageData);
end

################################################################################################
# Display Code Attribtes:
println("Server Name: ", server_name)
println("Data Name: ", dataname)
println("Mask Rate: ", maskId)
println("Sampling Type: ", usamp_type); println()

println("LANDMARK SELECTION ATTRIBUTES:")
println("No. of landmark points: ", noLandmark);
println("Method of landmark selection: ", ltype); println();

println("KERNEL GENERATION ATTRIBUTES")
dump(ktype); println()

println("DIMENSION REDUCTION ATTRIBUTES")
dump(dparams); println()

println("RECONSTRUCTION ATTRIBUTES:")
dump(param); println()

if usamp_type == "retrospective"
   println("The Error in reference to X* : ", err1); println()
   println("The Error in reference to DKB : ", err2); println()
end

################################################################################################
# Save the output files:
matwrite(filename, Dict(
    "Mlist" => Mlist,
    "noLandmark" => noLandmark,
    "ltype" => ltype,
    "noK" => noK,
    "d" => d,
    "Nfr" => Nfr,
    "lambda1" => λ1,
    "lambda2" => λ2,
    "lambda3" => λ3,
    "alpha" => alpha,
    "tau" => τ,
    "tauX" => τX,
    "tauZ" => τZ,
    "threshold" => threshold,
    "noIteration" => noIteration,
    "thresholdInner" => thresholdInner,
    "noIterationInner" => noIterationInner,
    "imgLoss" => imgLoss,
));
################################################################################################

