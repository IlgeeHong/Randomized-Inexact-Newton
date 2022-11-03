include("AugLsk.jl")
include("Eval.jl")

function GmresSQP(feature,label,con_A,con_b,Max_Iter,EPS_Res,eta1,eta2,delta,xi_B,beta,nu,rho)
    # data information
    N = size(feature)[1]
    nx = size(feature)[2]
    nlam = 11
    # Initialize
    k, X, Lam, NewDir, grad_eval, objcon_eval = 0, [ones(nx,)], [ones(nlam,)], zeros(nx+nlam), 0, 0
    # Evaluate objective, gradient, Hessian
    f_k = objec(feature, label, X[end])
    nabf_k = grad(feature, label, X[end])
    nab2f_k = Hess(feature, label, X[end])
    # Evaluate constraint and Jacobian
    c_k = con(con_A, con_b, X[end])
    G_k = Jac(con_A, con_b, X[end])
    grad_eval += 2
    objcon_eval += 2
    # Evaluate gradient and Hessian of Lagrangian
    nab_xL_k = nabf_k + G_k'Lam[end]
    nab_x2L_k = nab2f_k + Lam[end][end]*2*Diagonal(ones(nx))
    # KKT residual
    KKT = [norm([nab_xL_k; c_k])]
    # Indicator of convergence and singularity
    IdCon, IdSing = 1, 0
    # Other initial parameters
    Quant1, Quant2, Alpha = zeros(nx+nlam), zeros(nx+nlam), []
    Time = time()
    while KKT[end]>EPS_Res && k<Max_Iter
        # Compute Psi (upper bound of A inverse)
        xi_G = eigmin(Hermitian(Matrix(G_k*G_k'),:L))
        if xi_G < 1e-2
            IdSing = 1
            return [],[],[],[],grad_eval,objcon_eval,Time,0,IdSing
        else
            # Compute the reduced Hessian and do Hessian modification
            Q_k, R_k = qr(Matrix(G_k'))
            if nlam<nx && eigmin(Hermitian(Q_k[:,nlam+1:end]'nab_x2L_k*Q_k[:,nlam+1:end],:L)) < 1e-2
                 t_k = xi_B + norm(nab_x2L_k)
            else
                 t_k = 0
            end
            B_k = nab_x2L_k+t_k*Matrix(I,nx,nx)
            # Compute Psi (upper bound of KKT matrix inverse)
            psi = 7*(max(norm(B_k)^2,1)/(min(xi_G,1)*xi_B))
            # Compute Upsilon
            Upsilon = max(norm(B_k), norm(nab_x2L_k), norm(G_k))
            delta_trial = ((1/2-beta)*eta2)/(2*psi^2*(3*Upsilon+4*eta2*Upsilon^2+eta1*Upsilon^2))
            delta = min(delta, delta_trial)
            # Build KKT system
            A = hcat(vcat(nab_x2L_k+t_k*Matrix(I,nx,nx),G_k),vcat(G_k',zeros(nlam,nlam)))
            b = -vcat(nab_xL_k,c_k)
            # Initialize inexact direction
            NewDir_t = zeros(nx+nlam)
            r = b
            while true
                # Evaluate threshold
                Quant2 = delta*norm(b)/(norm(A)*psi)
                t = 0
                while norm(r) > Quant2 && t < 1000000
                    # GMRES
                    gmres!(NewDir_t,A,b,maxiter=1)
                    r = A*NewDir_t - b
                    t += 1
                end
                nabAugL_k = [nab_xL_k + eta2*nab2f_k*nab_xL_k + eta1*G_k'c_k; c_k+eta2*G_k*nab_xL_k]
                Quant1 = (nabAugL_k'NewDir_t)[1]
                if eta1 > 1e8 || eta2 <1e-8
                    NewDir = NewDir_t
                    break
                elseif Quant1 > -(eta2/2)*norm(b)^2
                    eta1 *= nu^2
                    eta2 *= (1/nu)
                    delta_trial = ((1/2-beta)*eta2)/(2*psi^2*(3*Upsilon+4*eta2*Upsilon^2+eta1*Upsilon^2)) ###
                    delta = min((delta/nu^4), delta_trial) ###
                else
                    NewDir = NewDir_t
                    break
                end
            end
            nabAugL_k = [nab_xL_k + eta2*nab2f_k*nab_xL_k + eta1*G_k'c_k; c_k+eta2*G_k*nab_xL_k]
            Quant1 = (nabAugL_k'NewDir)[1]
            AugL_k = f_k + c_k'Lam[end] + (eta1/2)*norm(c_k)^2 + (eta2/2)*norm(nab_xL_k)^2
            alpha_k = 1.0
            AugL_sk = AugLsk(feature,label,con_A,con_b,nx,X[end],Lam[end],eta1,eta2,alpha_k,NewDir)
            grad_eval += 2
            objcon_eval += 2
            # Armijo condition
            while AugL_sk > AugL_k + alpha_k*beta*Quant1
                alpha_k *= rho
                AugL_sk = AugLsk(feature,label,con_A,con_b,nx,X[end],Lam[end],eta1,eta2,alpha_k,NewDir)
                grad_eval += 2
                objcon_eval += 2
            end
            push!(X, X[end]+alpha_k*NewDir[1:nx])
            push!(Lam, Lam[end]+ alpha_k*NewDir[nx+1:end])
            push!(Alpha,alpha_k)
            k = k+1
            # Evaluate objective, gradient, Hessian
            f_k = objec(feature, label, X[end])
            nabf_k = grad(feature, label, X[end])
            nab2f_k = Hess(feature, label, X[end])
            # Evaluate constraint and Jacobian
            c_k = con(con_A, con_b, X[end])
            G_k = Jac(con_A, con_b, X[end])
            grad_eval += 2
            objcon_eval += 2
            # Evaluate gradient and Hessian of Lagrangian
            nab_xL_k = nabf_k + G_k'Lam[end]
            nab_x2L_k = nab2f_k + Lam[end][end]*2*Diagonal(ones(nx))
            push!(KKT, norm([nab_xL_k; c_k]))
            println(KKT[end])
        end
    end
    Time = time() - Time
    if k == Max_Iter
        return [],[],[],[],grad_eval,objcon_eval,Time,0,0
    else
        return X,Lam,KKT,Alpha,grad_eval,objcon_eval,Time,1,0
    end
end
