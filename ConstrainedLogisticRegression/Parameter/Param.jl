Aug = Parameter.AugParams(true,
                10000,  # Max_Iter
                1e-4,    # EPS_Res
                1.0,     # eta1
                0.1,     # eta2
                0.1,     # delta
                5,       # Rep
                0.1,     # xi_B
                0.1,     # beta
                1.4,     # nu
                0.5      # rho
                )

L1Adap = Parameter.L1AdapParams(true,
                10000,    # Max_Iter
                1e-4,    # EPS_Res
                1.0,     # penalty parameter
                0.1,     # kappa
                0.1,     # kappa1
                0.1,     # epsilon
                0.1,     # tau
                1e-8,     # eta
                5,       # Rep
                0.1      # xi_B
                )

L1 = Parameter.L1Params(true,
                10000,    # Max_Iter
                1e-4,    # EPS_Res
                1.0,     # penalty parameter
                1.0,     # kappa
                0.1,     # kappa1
                0.1,     # epsilon
                0.1,     # tau
                1e-8,     # eta 1e-8
                5,       # Rep
                0.1      # xi_B
                )
