include("GaussSketSQP.jl")

struct GaussSketResult
    XStep::Array
    LamStep::Array
    KKTStep::Array
    alpha_Step::Array
    Grad_eval::Array
    Objcon_eval::Array
    TimeStep::Array
end
## Implement GaussSketSQP for whole problem set
function GaussSketMain(Aug, Prob)
    Verbose = Aug.verbose
    Max_Iter = Aug.MaxIter
    EPS_Res = Aug.EPS_Res
    eta1 = Aug.eta1
    eta2 = Aug.eta2
    delta = Aug.delta
    TotalRep = Aug.Rep
    xi_B = Aug.xi_B
    beta = Aug.beta
    nu = Aug.nu
    rho = Aug.rho
    # Final result object
    GaussSketR = Array{GaussSketResult}(undef,length(Prob))
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
            println("GaussSketSQP","-",Idprob,"-",rep)
            X,Lam,KKT,Alpha,grad_eval,objcon_eval,Time,IdCon,IdSing = GaussSketSQP(nlp,Max_Iter,EPS_Res,eta1,eta2,delta,xi_B,beta,nu,rho)
            if IdSing == 1
                break
            elseif IdCon == 0
                println("Not convergent")
                rep += 1
                push!(XStep[rep], X)
                push!(LamStep[rep], Lam)
                push!(KKTStep[rep], KKT)
                push!(alpha_Step[rep], Alpha)
                push!(Grad_eval[rep], grad_eval)
                push!(Objcon_eval[rep], objcon_eval)
                push!(TimeStep[rep], Time)
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
        GaussSketR[Idprob] = GaussSketResult(XStep,LamStep,KKTStep,alpha_Step,Grad_eval,Objcon_eval,TimeStep)
        finalize(nlp)
    end
    return GaussSketR
end
