include("L1Lsk.jl")
include("model_reduction.jl")

function GaussSketL1SQP(nlp,Max_Iter,EPS_Res,mu,kappa,kappa1,epsilon,tau,eta,xi_B)
    nx = nlp.meta.nvar
    nlam = nlp.meta.ncon
    # Initialize
    k, X, Lam, NewDir, grad_eval, objcon_eval = 0, [nlp.meta.x0], [nlp.meta.y0], zeros(nx+nlam), 0, 0
    # evaluate objective, gradient, Hessian
    f_k, nabf_k = objgrad(nlp, X[end])
    nab2f_k = hess(nlp, X[end])
    # evaluate constraint and Jacobian
    c_k, G_k = consjac(nlp, X[end])
    grad_eval += 2
    objcon_eval += 2
    # evaluate gradient and Hessian of Lagrangian
    nab_xL_k = nabf_k + G_k'Lam[end]
    nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, Lam[end])
    # KKT residual
    KKT = [norm([nab_xL_k; c_k])]
    # Indicator of convergence and singularity
    IdCon, IdSing = 1, 0
    # Other initial parameters
    Quant1, Quant2, Alpha = zeros(nx+nlam), zeros(nx+nlam), []
    sigma = tau*(1.0-epsilon)
    beta = max(norm(nabf_k+G_k'Lam[end],1)/(norm(c_k,1)+1),1)
    kappa2 = beta
    Time = time()
    while KKT[end]>EPS_Res && k<Max_Iter
        # Compute lower bound of G_k(G_k)^T
        xi_G = eigmin(Hermitian(Matrix(G_k*G_k'),:L))
        if xi_G < 1e-6
            IdSing = 1
            return [NaN],[NaN],[NaN],[NaN],NaN,NaN,NaN,0,IdSing
        else
            # Compute the reduced Hessian and do Hessian modification
            Q_k, R_k = qr(Matrix(G_k'))
            if nlam<nx && eigmin(Hermitian(Q_k[:,nlam+1:end]'nab_x2L_k*Q_k[:,nlam+1:end],:L)) < 1e-6
                 t_k = xi_B + norm(nab_x2L_k)
            else
                 t_k = 0
            end
            B_k = nab_x2L_k+t_k*Matrix(I,nx,nx)
            # Build KKT system
            A = hcat(vcat(nab_x2L_k+t_k*Matrix(I,nx,nx),G_k),vcat(G_k',zeros(nlam,nlam)))
            b = -vcat(nab_xL_k,c_k)
            # Initialize inexact direction
            NewDir_t = zeros(nx+nlam)
            r = b
            Quant2 = kappa*norm(b,1)
            Quant3 = max(kappa1*norm(b[1:nx],1), kappa2*norm(b[nx+1:end],1))
            t = 0
            while t < 1000000
                # Gaussian random vector sketching
                S = rand(Normal(0,1),(nlam+nx,))
                NewDir_t -= ((S'*r)/(S'*A*A'*S))*(A'S)
                r = A*NewDir_t - b
                t += 1
                model_reduc = model_reduction(nabf_k,B_k,c_k,G_k,NewDir_t,mu,nx)
                Quant4 = sigma*mu*max(norm(c_k,1),norm(G_k*NewDir_t[1:nx]+c_k,1)-norm(c_k,1))
                # Termination test 1 and Termination test 2
                if (norm(r,1)<=Quant2 && norm(r[nx+1,end],1)<=Quant3 && model_reduc>=Quant4) || (norm(r[1:nx],1)<=epsilon*norm(c_k,1) && norm(r[nx+1:end],1)<=beta*norm(c_k,1)) #
                    NewDir = NewDir_t
                    break
                end
            end
            model_reduc = model_reduction(nabf_k,B_k,c_k,G_k,NewDir,mu,nx)
            Quant4 = sigma*mu*max(norm(c_k,1),norm(G_k*NewDir[1:nx]+c_k,1)-norm(c_k,1))
            # pass Termination test 2 but not model reduction
            if model_reduc < Quant4
                mu = max((nabf_k'NewDir[1:nx]+max((1/2)*NewDir[1:nx]'*B_k*NewDir[1:nx],0))[1]/(1-tau)/(norm(c_k,1)-norm(G_k*NewDir[1:nx]+c_k,1))[1], mu) + 1e-4
            end
            # directional derivative along inexact direction
            Quant1 = (nabf_k'NewDir[1:nx])[1]-mu*(norm(c_k,1)-norm(G_k*NewDir[1:nx]+c_k,1))[1]
            L1L_k = f_k + mu*norm(c_k,1)
            alpha_k = 1.0
            L1L_sk = L1Lsk(nlp,nx,X[end],mu,alpha_k,NewDir)
            objcon_eval += 2
            # Armijo condition
            while L1L_sk > L1L_k + alpha_k*eta*Quant1
                alpha_k *= 0.5
                L1L_sk = L1Lsk(nlp,nx,X[end],mu,alpha_k,NewDir)
                objcon_eval += 2
            end
            push!(X, X[end]+alpha_k*NewDir[1:nx])
            push!(Lam, Lam[end]+ alpha_k*NewDir[nx+1:end])
            push!(Alpha,alpha_k)
            k += 1
            f_k, nabf_k = objgrad(nlp, X[end])
            nab2f_k = hess(nlp, X[end])
            c_k, G_k = consjac(nlp, X[end])
            grad_eval += 2
            objcon_eval += 2
            nab_xL_k = nabf_k + G_k'Lam[end]
            nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, Lam[end])
            push!(KKT, norm([nab_xL_k; c_k]))
            println(KKT[end])
        end
    end
    Time = time() - Time
    if k == Max_Iter
        return [NaN],[NaN],[NaN],[NaN],NaN,NaN,NaN,0,0 
    else
        return X,Lam,KKT,Alpha,grad_eval,objcon_eval,Time,1,0
    end
end
