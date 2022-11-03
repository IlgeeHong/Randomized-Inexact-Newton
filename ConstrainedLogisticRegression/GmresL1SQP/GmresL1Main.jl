include("GmresL1SQP.jl")

struct GmresL1Result
    XStep::Array
    LamStep::Array
    KKTStep::Array
    alpha_Step::Array
    Grad_eval::Array
    Objcon_eval::Array
    TimeStep::Array
end
## Implement GmresSQP for whole problem set
# Gmres: parameters of Gmres sketching algorithm
# Prob: problem name set
function GmresL1Main(L1, Prob)
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
    GmresL1R = Array{GmresL1Result}(undef,length(Prob))
    ## Go over all Problems
    for Idprob = 1:length(Prob)
        # load problem
        println(Prob[Idprob])
        feature, label = load_dataset(string(Prob[Idprob]), dense = false, replace = false, verbose = true)
        N = size(feature)[1]
        nx = size(feature)[2]
        nlam = 11
        con_A = rand(Normal(0,1),(nlam-1, nx))
        con_b = rand(Normal(0,1),(nlam-1,))
        # define results vectors
        XStep = []
        LamStep = []
        KKTStep = []
        alpha_Step = []
        Grad_eval = []
        Objcon_eval = []
        TimeStep = []
        # go over all cases
        rep = 1
        while rep <= TotalRep
            println("GmresL1SQP","-",Idprob,"-",rep)
            X,Lam,KKT,Alpha,grad_eval,objcon_eval,Time,IdCon,IdSing = GmresL1SQP(feature,label,con_A,con_b,Max_Iter,EPS_Res,mu,kappa,kappa1,epsilon,tau,eta,xi_B)
            if IdSing == 1
                break
            elseif IdCon == 0
                println("Not convergent")
                rep += 1
            else
                push!(XStep, X)
                push!(LamStep, Lam)
                push!(KKTStep, KKT)
                push!(alpha_Step, Alpha)
                push!(Grad_eval, grad_eval)
                push!(Objcon_eval, objcon_eval)
                push!(TimeStep, Time)
                rep += 1
            end
        end
        GmresL1R[Idprob] = GmresL1Result(XStep,LamStep,KKTStep,alpha_Step,Grad_eval,Objcon_eval,TimeStep)
    end
    return GmresL1R
end
