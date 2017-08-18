# vim: ts=2 sw=2 et

module CausalMLTest
	#= if isdefined(:CausalML) =#
	#= 	reload("CausalML") =#
	#= end =#

  if ~("./" in LOAD_PATH)
    push!(LOAD_PATH, "./")
  end

	import CausalML
	reload("CausalML")
	using CausalML

	using Lbfgsb

  # Optional
	using Calculus
  #= using Convex =#
  #= using SCS =#
  using JLD

	const experiment_type = "binary"
	const p = 100
	const d = 10
	matrix_std = 0.8
	lambda = 1e-1
	n = Int32(1e4)

	epsilon = 1e-5
	#= run_ops = ["lbfgsb_test"] =# 
	#= run_ops = ["lbfgsb_test_2"] =# 
	#= run_ops = ["fast_lh_test"] =# 

	# Generate data
	println("Generating data")
	pop_data = PopulationData(p, d, matrix_std, experiment_type)
	emp_data = EmpiricalData(pop_data, n)

	#= if "lbfgsb_test" in run_ops =#
	function lbfgsb_test()
		dim = 2*p*(p-1)
		#= opt = Opt(:LD_LBFGS, dim) =#
		inner_fun(x, g) = lh(emp_data.sigmas_emp, emp_data, x, g)
		outer_fun(x, g) = l1_wrap(x, g, lambda, inner_fun)
		#= min_objective!(opt, outer_fun) =#
		#= lower_bounds!(opt, fill(0.0, dim)) =#
		lb = fill(0.0, dim)
		#xtol_abs!(opt, 1e-6)
		#= xtol_abs!(opt, rel_tol) =#

		B0 = zeros(p,p)
		B0 += 100 * triu(randn(p,p),1)
		start = mat2vec(B0, emp_data)
		start = [start; start]
		start[start .< 0] = 0
		(f, minx, numCall, numIter, status) = lbfgsb(outer_fun, start, lb = lb)
		B_lbfgsb = vec2mat(minx[1:p*(p-1)]-minx[p*(p-1)+1:dim], emp_data)
		println("LBFGSB, bad starting point: ", vecnorm(B_lbfgsb - pop_data.B))

		B0 = zeros(p,p)
		#= B0 += 10 * triu(ones(p,p),1) =#
		start = mat2vec(B0, emp_data)
		start = [start; start]
		start[start .< 0] = 0
		(f, minx, numCall, numIter, status) = lbfgsb(outer_fun, start, lb = lb)
		B_lbfgsb2 = vec2mat(minx[1:p*(p-1)]-minx[p*(p-1)+1:dim], emp_data)
		println("LBFGSB, good starting point: ", vecnorm(B_lbfgsb2 - pop_data.B))

#= 		if "l2_constr_test" in run_ops =#
#= 			x_base = zeros(length(start)) =#
#= 			upper_bound = norm(mat2vec(B))^2 =#
#= 			@time minx = solve_l2_constr(outer_fun, start, x_base, upper_bound, lb, fill(Inf, dim), 1e-7) =#
#= 			B_l2 = vec2mat(minx[1:p*(p-1)] - minx[p*(p-1)+1:dim]) =#
#= 			println("L2C: ", vecnorm(B_l2 - B)) =#
#= 		end =#
	end

	#= if "lbfgsb_test_2" in run_ops =#
	function lbfgsb_test_2()
		B0 = zeros(p, p)
		lh_data = VanillaLHData(p, lambda, B0)
		@time B_lbfgsb = min_vanilla_lh(emp_data, lh_data, low_rank = true)
		@time B_lbfgsb2 = min_vanilla_lh(emp_data, lh_data, low_rank = false)
		println("LBFGSB, good starting point: ", vecnorm(B_lbfgsb - pop_data.B))
		println("LBFGSB, difference: ", vecnorm(B_lbfgsb - B_lbfgsb2))

		B0 = zeros(p,p)
		B0 += 100 * triu(randn(p,p),1)
		lh_data.B0 = B0
		@time B_lbfgsb_bad = min_vanilla_lh(emp_data, lh_data, low_rank = true)
		println("LBFGSB, bad starting point: ", vecnorm(B_lbfgsb_bad - pop_data.B))
	end

	#= if "fast_lh_test" in run_ops =#
	function fast_lh_test()
		println("Testing likelihood")
		B0 = randn(p, p)
		B0[emp_data.diag_idx] = 0.0
		lh_data = VanillaLHData(p, lambda, B0)
		lh_data.B = B0
		#= println(lh_data.B) =#
		global g1 = zeros(p*(p-1))
		global g2 = zeros(p*(p-1))
		f(x) = lh(emp_data, lh_data, x, g1)
		#= global g3 = Calculus.gradient(f, mat2vec(B0, emp_data)) =#
		@time global l1 = lh(emp_data, lh_data, mat2vec(B0, emp_data), g1)
		@time global l2 = lh(emp_data, lh_data, mat2vec(B0, emp_data), g2, low_rank = true)
		#= println(lh_data.B) =#
		println(vecnorm(g1-g2))
		#= println("Gradient difference: ", vecnorm(g3 - g1)) =#
		B0 = randn(p, p)
		B0[emp_data.diag_idx] = 0.0
		#= lh_data2 = VanillaLHData(p, lambda, B0) =#
		#= lh_data2.B = B0 =#
		#= lh_data.B = B0 =#
		@time global l3 = lh(emp_data, lh_data, mat2vec(B0, emp_data), g1)
		@time global l4 = lh(emp_data, lh_data, mat2vec(B0, emp_data), g2, low_rank = true)
		println(l3)
		println(l4)
		println(vecnorm(g1-g2))
	end

	function l2_constr_test()
		B0 = zeros(p, p)
    lambda = 1e-3
		lh_data = VanillaLHData(p, lambda, B0)
		lh_data.x_base = mat2vec(pop_data.B, emp_data)
    lh_data.upper_bound = vecnorm(pop_data.B)^2
    lh_data.B0 = B0
		@time global B_lbfgsb = min_vanilla_lh(emp_data, lh_data, low_rank = true)
		global s
		global B_constr
		@time B_constr = min_constraint_lh(emp_data, lh_data)
		println("LBFGSB, difference: ", vecnorm(B_lbfgsb - pop_data.B))
		println("LBFGSB, constraint: ", vecnorm(B_constr - pop_data.B))
		println("Difference: ", vecnorm(B_constr - B_lbfgsb))
		#= println("Slack difference: ", s - vecnorm(B_constr-pop_data.B)^2) =#
		#= println("Slack: ", s) =#
	end

	#= function l2_constr_test_sticky() =#
    #= @load "sticky_data3.jld" =#
    #= lh_data.lambda = 1e4 =#
    #= lh_data.gamma = 1.5 =#
    #= #1= lh_data.x_base = zeros(p*(p-1)) =1# =#
	#= 	#1= B0 = zeros(p, p) =1# =#
    #= #1= lambda = 1e-3 =1# =#
	#= 	#1= lh_data = VanillaLHData(p, lambda, B0) =1# =#
	#= 	#1= lh_data.x_base = mat2vec(pop_data.B, emp_data) =1# =#
    #= #1= lh_data.upper_bound = vecnorm(pop_data.B)^2 =1# =#
    #= #1= lh_data.B0 = B0 =1# =#
	#= 	@time global B_lbfgsb = min_vanilla_lh(emp_data, lh_data, low_rank = true) =#
	#= 	global B_constr =#
	#= 	@time B_constr = min_constraint_lh(emp_data, lh_data) =#
	#= 	println("LBFGSB, difference: ", vecnorm(B_lbfgsb - pop_data.B)) =#
	#= 	println("LBFGSB, constraint: ", vecnorm(B_constr - pop_data.B)) =#
	#= 	println("Difference: ", vecnorm(B_constr - B_lbfgsb)) =#
	#= 	#1= println("Slack difference: ", s - vecnorm(B_constr-pop_data.B)^2) =1# =#
	#= 	#1= println("Slack: ", s) =1# =#
	#= end =#

	function quic_test()
    data = QuadraticPenaltyData(p)
    lambda = 1e0
    rho = 1.0
    global theta_prime = randn(p, p)
    theta_prime = theta_prime' * theta_prime
    data.lambda = lambda
    data.print_stats = true
		#= @time global theta1 = quic_old(p, emp_data, emp_data.sigmas_emp[1], rho = rho, lambda = lambda, print_stats = false) =#
		#= @time global theta2 = quic(emp_data, data, emp_data.sigmas_emp[1], rho = rho, print_stats = true, theta_prime = theta_prime) =#
    Profile.clear()
		#= @time global theta1 = quic_old(p, emp_data, emp_data.sigmas_emp[1], rho = rho, lambda = lambda) =#
		global theta2 = quic(emp_data, data, emp_data.sigmas_emp[1], rho = rho, theta_prime = theta_prime)
		println("Difference: ", vecnorm(theta2 - pop_data.thetas[1])^2)
    #= println("Inverse emp cov difference: ", vecnorm(inv(emp_data.sigmas_emp[1]) - pop_data.thetas[1])^2) =#
    #= println("Difference from each other: ", vecnorm(theta1 - theta2)) =#
    
    xvar = Variable(p, p)
    sigma = emp_data.sigmas_emp[1]
    theta_prime = eye(p)
    problem = minimize(sum(sigma .* xvar) - logdet(xvar) + lambda * vecnorm(xvar, 1) + rho/2 * vecnorm(xvar - theta_prime)^2)
    #= solve!(problem, SCSSolver()) =#
    #= x_convex = xvar.value =#

    #= println(vecnorm(x_convex - x_quic)) =#
    #= println(vecnorm(thetas[1] - x_convex)) =#
    #= println(vecnorm(thetas[1] - x_quic)) =#
    #= println(vecnorm(x_quic2 - x_fb)) =#
    #= println("Difference to convex program: ", vecnorm(theta2 - x_convex)) =#
    #= println(vecnorm(x_convex - x_fb)) =#
	end

	function admm_test()
		B0 = zeros(p, p)
    #= constr_data = ConstraintData_old(p) =#
    #= admm_data = ADMMData_old(emp_data, qu_data, constr_data) =#
    #= rho = 1.0 =#
    #= lambda = 1e-2 =#
    #= @time global B_admm = min_admm_old(emp_data, admm_data, lambda, B0, rho) =#
    #= println("ADMM, difference: ", vecnorm(B_admm - pop_data.B)) =#
    println(vecnorm(pop_data.B))

    constr_data = ConstraintData(p)
    lambda = 5e-3
    admm_data = ADMMData(emp_data, constr_data, lambda)
    admm_data.quic_data.tol = 1e-4
    rho = 1.0
    admm_data.rho = rho
    admm_data.quic_data.print_stats = false
    admm_data.quic_data.inner_tol = 1e-8
    admm_data.tol = 1e-1
    admm_data.low_rank = experiment_type == "single"
    @time global B_admm = min_admm(emp_data, admm_data, B0)
    println()
    #= println("Difference between ADMM results: ", vecnorm(B_admm - B_admm2)) =#
    println("ADMM, difference: ", vecnorm(B_admm- pop_data.B))
	end

	function admm_oracle_test()
		B0 = zeros(p, p)
    constr_data = ConstraintData(p)
    admm_data = ADMMData(emp_data, constr_data, 1.0)
    admm_data.tol = 1e-2
    admm_data.quic_data.print_stats = false
    admm_data.quic_data.tol = 1e-2
    admm_data.B0 = B0
    lambdas = logspace(-4, 4, 20)
    rho = 1.0
    admm_data.rho = rho
    admm_data.low_rank = experiment_type == "single"
    global errors
    (err, lambda, errors) = min_admm_oracle(pop_data, emp_data, admm_data, lambdas)
    println()
    #= println("Difference between ADMM results: ", vecnorm(B_admm - B_admm2)) =#
    println("ADMM, difference: ", err)
	end

	function lh_oracle_test()
		B0 = zeros(p, p)
    lambda = 1e-2
		lh_data = VanillaLHData(p, lambda, B0)
		#= lh_data.x_base = mat2vec(pop_data.B, emp_data) =#
    lh_data.x_base = mat2vec(zeros(p, p), emp_data)
    lh_data.upper_bound = vecnorm(pop_data.B)^2 # upper bound is squared
    lh_data.low_rank = experiment_type == "single"

    lambdas = flipdim(logspace(-4, 1, 10), 1)
    global errors
    global B_lh
    (B_lh, err, lambda, errors) = min_constr_lh_oracle(pop_data, emp_data, lh_data, lambdas)
    println()
    println("Constraint, difference: ", err)
	end

  function combined_oracle_test()
		B0 = zeros(p, p)
    constr_data = ConstraintData(p)
    admm_data = ADMMData(emp_data, constr_data, 1.0)
    admm_data.tol = 1e-2
    admm_data.quic_data.print_stats = false
    admm_data.quic_data.tol = 1e-6
    admm_data.B0 = B0
    rho = 1.0
    admm_data.rho = rho
    admm_data.low_rank = experiment_type == "single"
		lh_data = VanillaLHData(p, lambda, B0)
    lh_data.low_rank = experiment_type == "single"

    lambdas = logspace(-1, 1, 5)

    global errors1, errors2, B1, B2
    (B1, B2, err1, err2, lambda1, lambda2, errors1, errors2) = combined_oracle(pop_data, emp_data, admm_data, lh_data, lambdas)

    println()
    #= println("Difference between ADMM results: ", vecnorm(B_admm - B_admm2)) =#
    println("ADMM, difference: ", err1)
    println("LH, difference: ", err2)
  end

  function combined_oracle_screen()
    ps = [100]
    #= ps = [10] =#
    #= ns = logspace(log10(50), 4, 10) =#
    ns = 1000
    #= ds = union(1:10, 12:2:20, 24:4:40) =#
    #= ds = 1:10 =#
    #= ds = [5,6] =#
    ds = 5
    trials = 2
    global combined_results = []
    #= lambdas = flipdim(logspace(-4, -1, 40), 1) =#
    #= lambdas = flipdim(logspace(-4, -1, 50), 1) =#
    lambdas = [1e-3]
    #= lambdas = logspace(-4, -1, 3) =#
    #= lambdas = [1e-2] =#

    if length(ARGS) >= 1 
      suffix = ARGS[1]
    else
      suffix = ""
    end

    for n in ns
      for p in ps
        for d in ds
          errors1_trials = zeros(trials)
          lambdas1_trials = zeros(trials)
          errors2_trials = zeros(trials)
          lambdas2_trials = zeros(trials)
          errors_llc_trials = zeros(trials)
          lambdas_llc_trials = zeros(trials)
          ground_truth_norms = zeros(trials)
          for trial in 1:trials
            println("Generating data")
            pop_data = PopulationData(p, d, matrix_std, experiment_type)
            emp_data = EmpiricalData(pop_data, n)
            ground_truth_norms[trial] = vecnorm(pop_data.B)

            B0 = zeros(p, p)
            constr_data = ConstraintData(p)
            admm_data = ADMMData(emp_data, constr_data, 1.0)
            admm_data.tol_abs = 5e-4
            admm_data.tol_rel = 1e-3
            admm_data.quic_data.print_stats = false
            admm_data.quic_data.tol = 1e-6
            admm_data.dual_balancing = true
            admm_data.B0 = B0
            rho = 50.0
            admm_data.rho = rho
            admm_data.low_rank = experiment_type == "single"
            lh_data = VanillaLHData(p, 1, B0)
            lh_data.low_rank = experiment_type == "single"
            lh_data.final_tol = 1e-3

            global errors1, errors2, B1, B2
            (B1, B2, err1, err2, lambda1, lambda2, errors1, errors2) = combined_oracle(pop_data, emp_data, admm_data, lh_data, lambdas)
            (B, err, lambda, _) = llc(pop_data, emp_data, lambdas)
            errors1_trials[trial] = err1
            errors2_trials[trial] = err2
            errors_llc_trials[trial] = err
            lambdas1_trials[trial] = lambda1
            lambdas2_trials[trial] = lambda2
            lambdas_llc_trials[trial] = lambda

            println()
            #= println("Difference between ADMM results: ", vecnorm(B_admm - B_admm2)) =#
            println("ADMM, difference: ", err1)
            println("LH, difference: ", err2)
          end

          push!(combined_results, Dict("n"=>n, "p"=>p, "d"=>d,
                                       "errs1"=> errors1_trials,
                                       "errs2"=>errors2_trials,
                                       "errs_llc"=>errors_llc_trials,
                                       "lambdas1"=>lambdas1_trials,
                                       "lambdas2"=>lambdas2_trials,
                                       "lambdas_llc"=>lambdas_llc_trials,
                                       "gt"=>ground_truth_norms))

          fname = "results4_norm_vard_" * suffix * ".jld"
          #= jldopen(fname, "w") do file =#
          open(fname, "w") do file
            serialize(file, combined_results)
            #= write(file, "combined_results", combined_results) =#
          end
          #= save(fname, "combined_results", combined_results) =#
        end
      end
    end
  end

  function combined_llc_screen()
    ps = [100]
    #= ds = union(1:10, 12:2:20, 24:4:40) =#
    #= ds = 1:10 =#
    ds = 5
    trials = 3
    global combined_results_llc = []
    lambdas = logspace(-4, -1, 30)
    for p in ps
      for d in ds
        errors_trials = zeros(trials)
        lambdas_trials = zeros(trials)
        for trial in 1:trials
          println("Generating data")
          pop_data = PopulationData(p, d, matrix_std, experiment_type)
          emp_data = EmpiricalData(pop_data, n)

          (B, err, lambda, _) = llc(pop_data, emp_data, lambdas)
          errors_trials[trial] = err
          lambdas_trials[trial] = lambda

          println()
        end

        push!(combined_results_llc, Dict("p"=>p, "d"=>d, "errs"=> errors_trials,
                                    "lambdas"=>lambdas_trials,
                                   ))

        #= @save "results2.jld" combined_results_llc =#
      end
    end
  end

  function llc_test()
    lambdas = logspace(-4, 0, 15)
    global B_llc, errors
    (B_llc, err, lambda, errors) = llc(pop_data, emp_data, lambdas)
    println("LLC, difference: ", vecnorm(B_llc - pop_data.B))
  end

	#= fast_lh_test() =#
	#= lbfgsb_test_2() =#
	#= quic_test() =#
  #= admm_oracle_test() =#
  #= admm_test() =#
  #= llc_test() =#
  #= lh_oracle_test() =#
  #= combined_oracle_test() =#
  @time combined_oracle_screen()
  #= combined_llc_screen() =#
	#= l2_constr_test() =#
	#= l2_constr_test_sticky() =#
end
