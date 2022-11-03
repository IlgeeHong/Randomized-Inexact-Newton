include("GaussSketL1AdapSQP.jl")

struct GaussSketL1AdapResult
    XStep::Array
    LamStep::Array
    KKTStep::Array
    alpha_Step::Array
    Grad_eval::Array
    Objcon_eval::Array
    TimeStep::Array
end
## Implement GaussSketSQP for whole problem set
function GaussSketL1AdapMain(L1Adap, Prob)
    Verbose = L1Adap.verbose
    Max_Iter = L1Adap.MaxIter
    EPS_Res = L1Adap.EPS_Res
    mu = L1Adap.mu
    kappa = L1Adap.kappa
    kappa1 = L1Adap.kappa1
    epsilon = L1Adap.epsilon
    tau = L1Adap.tau
    eta = L1Adap.eta
    TotalRep = L1Adap.Rep
    xi_B = L1Adap.xi_B
    # Final result object
    GaussSketL1AdapR = Array{GaussSketL1AdapResult}(undef,length(Prob))
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
            println("GaussSketL1AdapSQP","-",Idprob,"-",rep)
            X,Lam,KKT,Alpha,grad_eval,objcon_eval,Time,IdCon,IdSing = GaussSketL1AdapSQP(nlp,Max_Iter,EPS_Res,mu,kappa,kappa1,epsilon,tau,eta,xi_B)
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
        GaussSketL1AdapR[Idprob] = GaussSketL1AdapResult(XStep,LamStep,KKTStep,alpha_Step,Grad_eval,Objcon_eval,TimeStep)
        finalize(nlp)
    end
    return GaussSketL1AdapR
end
