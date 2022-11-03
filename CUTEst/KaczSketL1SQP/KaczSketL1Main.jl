include("KaczSketL1SQP.jl")

struct KaczSketL1Result
    XStep::Array
    LamStep::Array
    KKTStep::Array
    alpha_Step::Array
    Grad_eval::Array
    Objcon_eval::Array
    TimeStep::Array
end
## Implement KaczSketSQP for whole problem set
# KaczSket: parameters of Kacz sketching algorithm
# Prob: problem name set
function KaczSketL1Main(L1, Prob)
    Verbose = L1.verbose
    Max_Iter = L1.MaxIter
    EPS_Res = L1.EPS_Res
    mu = L1.mu
    kappa = L1.kappa
    kappa1 = L1.kappa1
    epsilon = L1.epsilon
    tau = L1.tau
    eta = L1.eta
    TotalRep = L1.Rep
    xi_B = L1.xi_B
    # Final result object
    KaczSketL1R = Array{KaczSketL1Result}(undef,length(Prob))
    ## Go over all Problems
    for Idprob = 1:length(Prob)
        # load problem
        println(Prob[Idprob])
        nlp = CUTEstModel(Prob[Idprob])
        # define results vectors
        XStep = reshape([[] for i = 1:TotalRep], (1,TotalRep))
        LamStep = reshape([[] for i = 1:TotalRep], (1,TotalRep))
        KKTStep = reshape([[] for i = 1:TotalRep], (1,TotalRep))
        alpha_Step = reshape([[] for i = 1:TotalRep], (1,TotalRep))
        Grad_eval = reshape([[] for i = 1:TotalRep], (1,TotalRep))
        Objcon_eval = reshape([[] for i = 1:TotalRep], (1,TotalRep))
        TimeStep = reshape([[] for i = 1:TotalRep], (1,TotalRep))
        # go over all cases
        rep = 1
        while rep <= TotalRep
            println("KaczSketL1SQP","-",Idprob,"-",rep)
            X,Lam,KKT,Alpha,grad_eval,objcon_eval,Time,IdCon,IdSing = KaczSketL1SQP(nlp,Max_Iter,EPS_Res,mu,kappa,kappa1,epsilon,tau,eta,xi_B)
            if IdSing == 1
                break
            elseif IdCon == 0
                println("Not convergent")
                push!(XStep[rep], X)
                push!(LamStep[rep], Lam)
                push!(KKTStep[rep], KKT)
                push!(alpha_Step[rep], Alpha)
                push!(Grad_eval[rep], grad_eval)
                push!(Objcon_eval[rep], objcon_eval)
                push!(TimeStep[rep], Time)
                rep += 1
            else
                push!(XStep[rep], X)
                push!(LamStep[rep], Lam)
                push!(KKTStep[rep], KKT)
                push!(alpha_Step[rep], Alpha)
                push!(Grad_eval[rep], grad_eval)
                push!(Objcon_eval[rep], objcon_eval)
                push!(TimeStep[rep], Time)
                rep += 1
            end
        end
        KaczSketL1R[Idprob] = KaczSketL1Result(XStep,LamStep,KKTStep,alpha_Step,Grad_eval,Objcon_eval,TimeStep)
        finalize(nlp)
    end
    return KaczSketL1R
end
