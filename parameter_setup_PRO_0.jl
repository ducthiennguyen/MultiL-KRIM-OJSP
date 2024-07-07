################################################################################################
## Parameters for Landmark Extraction (Step 1):
noLandmark = [
    150
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
kdict = [
    ("gaussian", kparams1),
    ("gaussian", kparams2),
    ("gaussian", kparams3),
    ("polynomial", kparams4),
    ("polynomial", kparams5),
    ("polynomial", kparams6),
    ("polynomial", kparams7)
];

kernel = [3];
ktype = kdict[kernel];

diffType = "diff";

################################################################################################
## Dimension Reduction using RSE:
"""
Parameters to be used:
d
"""
λw = 1e-3; 
αw = 0.5;
d = 10;
threshold = 1e-4;
noIteration = 1e5;
dparams = dimRedParams(λw, αw, d, threshold, noIteration);

################################################################################################
## Reconstruction framework:
"""
Parameters to be used:
λL (the sobolev smoothness term), alpha (lambda4 in paper), λ1, τ, τX
"""

ζ = 1e-3;
γ = 0.9;
λ1 = repeat([0.5], 5);
λ2 = 1e0;
λ3 = 0.05;
λs = 0.5;
λn = 1e-2;
μn = 4e-2;
λL = 20;
eps = 0;
λD = 0.5;
λX = 2.3;
ϵ = 1e0;
C = repeat([1e2], 5);
alpha = repeat([0.01], 5);
CA = 5e0;
CG = 2e0;
τ = repeat([1e-5], 5);
τX = 2.5;
τZ = 1e-10;
τA = 1e-10;
τG = 1e-10;
τB = 1e-10;
τS = 1e-10;
α = 0.5;
αA = 0.5;
αG = 0.5;
αB = 0.5;
αS = 0.5;
αX = 0.5;
noK = length(ktype);
dList = repeat(noLandmark, noK);
Nl = reduce(+, dList);
threshold = 1e-6;
noIteration = 10000;
thresholdInner = 1e-5;
noIterationInner = 1000;
param = OptimizerParams(N, T, ζ, γ, λ1, λ2, λ3, λs, λn, μn, λL, ϵ, C, alpha, CA, CG, τ, τA, τG, τB, τS, α, αA, αG, αB, αS, noK, threshold, noIteration, thresholdInner, noIterationInner);
################################################################################################
