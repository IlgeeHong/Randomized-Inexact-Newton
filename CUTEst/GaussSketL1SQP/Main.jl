using NLPModels
using LinearOperators
using OptimizationProblems
using MathProgBase
using ForwardDiff
using CUTEst
using NLPModelsJuMP
using LinearAlgebra
using Distributed
using Ipopt
using DataFrames
using PyPlot
using MATLAB
using Glob
using DelimitedFiles
using Random
using Distributions

cd("/Users/ilgeehong/Desktop/SQPwithSketching/CUTEst/GaussSketL1SQP")
# read problem
Prob = readdlm(string(pwd(),"/../Parameter/problems.txt"))

# define parameter module
module Parameter
    struct AugParams
        verbose                            # Do we create dump dir?
        # stopping parameters
        MaxIter::Int                       # Maximum Iteration
        EPS_Res::Float64                   # minimum of difference
        # adaptive parameters
        eta1::Float64                      # eta1
        eta2::Float64                      # eta2
        delta::Float64                     # delta
        # fixed parameters
        Rep::Int                           # Number of Independent runs
        xi_B::Float64                      # xi_B
        beta::Float64                      # beta
        nu::Float64                        # nu
        rho::Float64                       # rho
    end
    struct L1Params
        verbose                            # Do we create dump dir?
        # stopping parameters
        MaxIter::Int                       # Maximum Iteration
        EPS_Res::Float64                   # minimum of difference
        # fixed parameters
        mu::Float64                        # mu
        kappa::Float64                     # kappa
        kappa1::Float64                    # kappa1
        epsilon::Float64                   # epsilon
        tau::Float64                       # tau
        eta::Float64                       # delta
        Rep::Int
        xi_B::Float64                     # delta
    end
    struct L1AdapParams
        verbose                            # Do we create dump dir?
        # stopping parameters
        MaxIter::Int                       # Maximum Iteration
        EPS_Res::Float64                   # minimum of difference
        # fixed parameters
        mu::Float64                        # mu
        kappa::Float64                     # kappa
        kappa1::Float64                    # kappa1
        epsilon::Float64                   # epsilon
        tau::Float64                       # tau
        eta::Float64                       # delta
        Rep::Int
        xi_B::Float64                     # delta
    end
end

using Main.Parameter
include("GaussSketL1Main.jl")

#######################################
#########  run main file    ###########
#######################################
function main()
    Random.seed!(2023)
    ## run GaussSket SQP
    # load parameter value
    include("../Parameter/Param.jl")
    # add parameter value and problem
    GaussSketL1R = GaussSketL1Main(L1, Prob)
    if L1.verbose
        NumProb = 47
        decom = convert(Int64, floor(length(GaussSketL1R)/NumProb))
        for i=1:decom
            path = string("../Solution/GaussSketL1SQP", i, ".mat")
            Result = GaussSketL1R[(i-1)*NumProb+1:i*NumProb]
            write_matfile(path; Result)
        end
        path = string("../Solution/GaussSketL1R", decom+1, ".mat")
        Result = GaussSketL1R[decom*NumProb+1:end]
        write_matfile(path; Result)
    end
end

main()
