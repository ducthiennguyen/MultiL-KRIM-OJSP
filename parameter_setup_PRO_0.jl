################################################################################################
## Parameters for Landmark Extraction (Step 1):
noLandmark = [
    70
];
ltype = "maxmin";

################################################################################################
## Initializing the kernel parameters for Kernel construction:
kparams1 = kernelParams(); kparams1.alpha = 0.2;
kparams2 = kernelParams(); kparams2.alpha = 0.4;
kparams3 = kernelParams(); kparams3.alpha = 0.8;
kparams4 = kernelParams(); kparams4.degree = 1; kparams4.intercept = 0.0;
kparams5 = kernelParams(); kparams5.degree = 2; kparams5.intercept = 0.0;
kparams6 = kernelParams(); kparams6.degree = 3; kparams6.intercept = 0.0;
kparams7 = kernelParams(); kparams7.degree = 4; kparams7.intercept = 0.0;
ktype = [
    ("gaussian", kparams3),
];

################################################################################################
## Dimension Reduction using RSE:
"""
for multil-KRIM, only d is used
"""
λw = 1e-3; 
αw = 0.5;
d = 4;
threshold = 1e-4;
noIteration = 1e5;
dparams = dimRedParams(λw, αw, d, threshold, noIteration);

################################################################################################
## Reconstruction framework:
"""
change Nfr to select the first Nfr frames
most influential params seem to be λ1, (λ3/λ2)
alpha can also be changed (for quadratic Regularization)
τ, τX can also be changed, usually it's not important

other things are not important/used
"""
nCycles = 15;
nPhases = 24;
Nfr = nCycles*nPhases; # MRXCAT data has 15 cycles x 24 phases (see paper)

ζ = 1e-3;
γ = 0.9;
λ1 = repeat([0.1], 10);
λ2 = 5e-1;
λ3 = 1e0; 
λs = 5e-1;
λn = 1e-1;
μn = 4e-2;
λL = 1e-1;
ϵ = 1e0;
alpha = repeat([0.1], 10);
τ = repeat([1e-5], 10);
τX = 1e-5;
τZ = τX;
τS = 5e-1;
α = 0.5;
αS = 0.5;
noK = length(ktype);
dList = repeat(noLandmark, noK);
Nl = reduce(+, dList);
threshold = 1e-5;
noIteration = 7000;
thresholdInner = 1e-5;
noIterationInner = 300;
param = OptimizerParams(0, 0, Nfr, ζ, γ, λ1, λ2, λ3, λs, λn, μn, λL, ϵ, alpha, τ, τS, α, αS, noK, threshold, noIteration, thresholdInner, noIterationInner);
################################################################################################
