# vim: ts=2 sw=2 et

module CausalMLTest
	#= if isdefined(:CausalML) =#
	#= 	reload("CausalML") =#
	#= end =#

  function loadvar(fname)
    open(fname)	do file
      deserialize(file)
    end
  end

  if ~("./" in LOAD_PATH)
    push!(LOAD_PATH, "./")
  end
  if ~("./CausalML/" in LOAD_PATH)
    push!(LOAD_PATH, "./CausalML/")
  end

	import CausalML
	reload("CausalML")
	using CausalML

	using Lbfgsb

  # Optional
	using Calculus
  using LightGraphs
  using Convex
  using SCS
  #= using JLD =#

	experiment_type = "binary"
	const p = 50
	const d = 5
	matrix_std = 5.0
	lambda = 1e-1
	n = Int32(1e4)

	epsilon = 1e-5
	#= run_ops = ["lbfgsb_test"] =# 
	#= run_ops = ["lbfgsb_test_2"] =# 
	#= run_ops = ["fast_lh_test"] =# 

	# Generate data
  function gen_data()
    println("Generating data")
    while true
      global pop_data = PopulationData(p, d, matrix_std, experiment_type)
      global emp_data = EmpiricalData(pop_data, n)
      if maximum(cond.(pop_data.thetas)) > 1e4 #|| true
        break
      end
    end
  end

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
    println(abs(l1-l2))
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
    println(abs(l3-l4))
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

	function graph_lasso_admm_test()
    gen_data()
    data = QuadraticPenaltyData(p)
    lambda = 1e-2
    rho = 10.0
    B0 = 10*triu(randn(p, p), 1)
    #= B0 = zeros(p, p) =#
    U = emp_data.Us[2]
    global theta_prime = (I - U * B0)' * (I - U * B0)
    #= theta_prime = zeros(p,p) =#
    data.lambda = lambda
    data.print_stats = false
    data.inner_mult = 1/3
    data.inner_min_iterations = 10
    data.inner_tol = 1e-7
    data.max_iterations = 50
    data.inner_max_iterations = 2000
    data.relaxation = 0.2
    data.tol = 1e-7
    data.hard_thresh = 1e-10
    data.dual_balancing = true
    #= data.balance_mult = =# 

    #= global theta0 = inv(emp_data.sigmas_emp[2]) =#
    #= symmetrize!(theta0) =#
    theta0 = zeros(p, p)
    #= data.theta0 = theta0 =#
    #= println(theta0) =#
		#= @time global theta1 = quic_old(p, emp_data, emp_data.sigmas_emp[1], rho = rho, lambda = lambda, print_stats = false) =#
		#= @time global theta2 = quic(emp_data, data, emp_data.sigmas_emp[1], rho = rho, print_stats = true, theta_prime = theta_prime) =#
    Profile.clear()
		#= @time global theta1 = quic_old(p, emp_data, emp_data.sigmas_emp[1], rho = rho, lambda = lambda) =#
    println("Starting quic")
    (theta_quic, status) = quic(emp_data, data, emp_data.sigmas_emp[2], rho = rho, theta_prime = theta_prime)
    #= @time (theta_quic, status) = quic(emp_data, data, emp_data.sigmas_emp[2], rho = rho, theta_prime = theta_prime) =#
    theta_quic = copy(theta_quic)

    #= data.tol = 1e-5 * 0.1 =#
    #= data.tol_rel = 1e-4 * 0.1 =#
    data.max_iterations = 1000
    global theta0 = inv(emp_data.sigmas_emp[2])
    symmetrize!(theta0)
    (theta_admm, status) = graph_lasso_admm(emp_data, data, emp_data.sigmas_emp[2], rho = rho, theta_prime = theta_prime)
    @time (theta_admm, status) = graph_lasso_admm(emp_data, data, emp_data.sigmas_emp[2], rho = rho, theta_prime = theta_prime)
		println("Difference admm: ", vecnorm(theta_admm - pop_data.thetas[2])^2)
    #= println("Inverse emp cov difference: ", vecnorm(inv(emp_data.sigmas_emp[1]) - pop_data.thetas[1])^2) =#
    #= println("Difference from each other: ", vecnorm(theta1 - theta2)) =#
    
    xvar = Variable(p, p)
    sigma = emp_data.sigmas_emp[2]
    #= theta_prime = eye(p) =#
    problem = minimize(sum(sigma .* xvar) - logdet(xvar) + lambda * vecnorm(xvar, 1) + rho/2 * vecnorm(xvar - theta_prime)^2)
    #= solve!(problem, SCSSolver(max_iters = 10000)) =#
    #= global x_convex = xvar.value =#
    x_convex = eye(p)
    

    function g_min(theta)
      theta_inv = inv(theta)
      symmetrize!(theta_inv)
      G = sigma - theta_inv + rho * (theta - theta_prime)
      for i in eachindex(G)
        if theta[i] > 0
          G[i] += lambda
        elseif theta[i] < 0
          G[i] -= lambda
        else
          G[i] = soft_thresh(G[i], lambda)
        end
      end
      return G
    end

    println("Difference to convex, admm: ", vecnorm(x_convex - theta_admm)/vecnorm(x_convex))
    println("Difference to convex, quic: ", vecnorm(x_convex - theta_quic)/vecnorm(x_convex))
    objective(theta) = sum(sigma .* theta) - logdet(theta) + lambda * vecnorm(theta, 1) + rho/2 * vecnorm(theta - theta_prime)^2
    println("Objectives: convex = ", objective(x_convex))
    println("Objectives: admm = ", objective(theta_admm))
    println("Objectives: quic = ", objective(theta_quic))
    println("Gmins: admm = ", vecnorm(g_min(theta_admm)), ", quic = ", vecnorm(g_min(theta_quic)))
	end

	function quic_test()
    gen_data()
    data = QuadraticPenaltyData(p)
    lambda = 1e-2
    rho = 0.0
    #= B0 = 10*triu(randn(p, p), 1) =#
    B0 = zeros(p, p)
    U = emp_data.Us[2]
    global theta_prime = (I - U * B0)' * (I - U * B0)
    #= theta_prime = zeros(p,p) =#
    data.lambda = lambda
    data.print_stats = true
    data.inner_mult = 1/3
    data.inner_min_iterations = 5
    data.inner_tol = 1e-6
    data.max_iterations = 10000
    data.inner_max_iterations = 10000
    data.relaxation = 0.2
    data.tol = 1e-8
    data.hard_thresh = 1e-10

    global theta0 = inv(emp_data.sigmas_emp[2])
    symmetrize!(theta0)
    #= data.theta0 = theta0 =#
    #= println(theta0) =#
		#= @time global theta1 = quic_old(p, emp_data, emp_data.sigmas_emp[1], rho = rho, lambda = lambda, print_stats = false) =#
		#= @time global theta2 = quic(emp_data, data, emp_data.sigmas_emp[1], rho = rho, print_stats = true, theta_prime = theta_prime) =#
    Profile.clear()
		#= @time global theta1 = quic_old(p, emp_data, emp_data.sigmas_emp[1], rho = rho, lambda = lambda) =#
    global theta2
		theta2, status = quic(emp_data, data, emp_data.sigmas_emp[2], rho = rho, theta_prime = theta_prime)
		println("Difference: ", vecnorm(theta2 - pop_data.thetas[1])^2)
    #= println("Inverse emp cov difference: ", vecnorm(inv(emp_data.sigmas_emp[1]) - pop_data.thetas[1])^2) =#
    #= println("Difference from each other: ", vecnorm(theta1 - theta2)) =#
    
    xvar = Variable(p, p)
    sigma = emp_data.sigmas_emp[2]
    #= theta_prime = eye(p) =#
    problem = minimize(sum(sigma .* xvar) - logdet(xvar) + lambda * vecnorm(xvar, 1) + rho/2 * vecnorm(xvar - theta_prime)^2)
    solve!(problem, SCSSolver())
    global x_convex = xvar.value

    println(vecnorm(x_convex - theta2)/vecnorm(x_convex))
    objective(theta) = sum(sigma .* theta) - logdet(theta) + lambda * vecnorm(theta, 1) + rho/2 * vecnorm(theta - theta_prime)^2
    println("Objectives: ", objective(x_convex), ", ", objective(theta2))
    #= println(vecnorm(x_convex - x_quic)) =#
    #= println(vecnorm(thetas[1] - x_convex)) =#
    #= println(vecnorm(thetas[1] - x_quic)) =#
    #= println(vecnorm(x_quic2 - x_fb)) =#
    #= println("Difference to convex program: ", vecnorm(theta2 - x_convex)) =#
    #= println(vecnorm(x_convex - x_fb)) =#
	end

	function admm_test()
    gen_data()
		#= B0 = zeros(p, p) =#
    #= constr_data = ConstraintData_old(p) =#
    #= admm_data = ADMMData_old(emp_data, qu_data, constr_data) =#
    #= rho = 1.0 =#
    #= lambda = 1e-2 =#
    #= @time global B_admm = min_admm_old(emp_data, admm_data, lambda, B0, rho) =#
    #= println("ADMM, difference: ", vecnorm(B_admm - pop_data.B)) =#

    constr_data = ConstraintData(p)
    lambda = 1e-2
    #= global B0 = 10 * triu(randn(p, p), 1) =#
    global B0 = zeros(p, p)
    global pop_data

    admm_data = ADMMData(emp_data, constr_data, 1.0)

    admm_data.tol_abs = 5e-4
    admm_data.tol_rel = 1e-3
    admm_data.quic_data.print_stats = false
    admm_data.quic_data.tol = 1e-6
    admm_data.quic_data.inner_tol = 1e-4
    admm_data.quic_data.inner_min_iterations = 20
    admm_data.quic_data.inner_max_iterations = 1000
    admm_data.quic_data.inner_mult = 1/3
    admm_data.quic_data.hard_thresh = 1e-10
    admm_data.quic_data.relaxation = 0.2
    admm_data.quic_data.beta = 0.5
    admm_data.quic_data.dual_balancing = true
    admm_data.dual_balancing = true
    admm_data.bb = false
    admm_data.tighten = false
    admm_data.mu = 4
    admm_data.tau = 1.5
    admm_data.B0 = B0
    admm_data.quic_data.max_iterations = 1000
    rho = 1.0

    lambdas = flipdim(logspace(-4, -1, 20), 1)
    admm_data.rho = rho
    admm_data.low_rank = experiment_type != "binary"

    global B_admm
    @time (B_admm, status) = min_admm(emp_data, admm_data)
    println()
    #= println("Difference between ADMM results: ", vecnorm(B_admm - B_admm2)) =#
    println("ADMM, difference: ", vecnorm(B_admm- pop_data.B))
    println("Status: ", status)
	end

	function admm_oracle_test()
    gen_data()
		#= global B0 = zeros(p, p) =#
    global B0 = 10 * triu(randn(p, p), 1)
    constr_data = ConstraintData(p)
    admm_data = ADMMData(emp_data, constr_data, 1.0)

    admm_data.tol_abs = 5e-4
    admm_data.tol_rel = 1e-3
    admm_data.quic_data.print_stats = false
    admm_data.quic_data.tol = 1e-6
    admm_data.quic_data.inner_tol = 1e-6
    admm_data.quic_data.inner_min_iterations = 20
    admm_data.quic_data.inner_mult = 1/3
    admm_data.quic_data.hard_thresh = 1e-10
    admm_data.quic_data.relaxation = 0.2
    admm_data.quic_data.beta = 0.5
    admm_data.dual_balancing = true
    admm_data.bb = false
    admm_data.tighten = false
    admm_data.mu = 4
    admm_data.tau = 1.5
    admm_data.B0 = B0
    admm_data.quic_data.max_iterations = 150
    rho = 1.0

    lambdas = flipdim(logspace(-4, 0, 20), 1)
    admm_data.rho = rho
    admm_data.low_rank = experiment_type != "binary"
    global errors, B_admm
    (B_admm, err, lambda, errors) = min_admm_oracle(pop_data, emp_data, admm_data, lambdas)
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
    lh_data.low_rank = experiment_type != "binary"

    lambdas = flipdim(logspace(-4, 1, 10), 1)
    global errors
    global B_lh
    (B_lh, err, lambda, errors) = min_constr_lh_oracle(pop_data, emp_data, lh_data, lambdas)
    println()
    println("Constraint, difference: ", err)
	end

  function combined_oracle_test()
    experiment_type = "binary"
    n = 2000
    p = 50
    d = 5
    k = 2
    lambdas = flipdim(logspace(-4, 0, 30), 1)

    println("Generating data")
    global pop_data = PopulationData(p, d, matrix_std, experiment_type)

    emp_data = EmpiricalData(pop_data, n)

    B0 = zeros(p, p)
    constr_data = ConstraintData(p)
    admm_data = ADMMData(emp_data, constr_data, 1.0)
    admm_data.tol_abs = 5e-4
    admm_data.tol_rel = 1e-3
    admm_data.quic_data.print_stats = false
    admm_data.quic_data.tol = 1e-6
    admm_data.dual_balancing = true
    admm_data.bb = false
    admm_data.tighten = true
    admm_data.mu = 500
    admm_data.tau = 1.5
    admm_data.B0 = B0
    rho = 1.0
    #= rho = 0.8670764957916309 =#
    admm_data.rho = rho
    admm_data.low_rank = experiment_type != "binary"
		lh_data = VanillaLHData(p, lambda, B0)
    lh_data.low_rank = experiment_type != "binary"

    lambdas = logspace(-4, 0, 30)

    global errors1, errors2, B1, B2
    (B1_oracle, B2_oracle, err1, err2, lambda1_oracle, lambda2_oracle, errors1, errors2) = combined_oracle(pop_data, emp_data, admm_data, lh_data, lambdas)
    (B1_cv, B2_cv, lh1, lh2, lambda1_cv, lambda2_cv, lhs1, lhs2) = combined_cv(emp_data, admm_data, lh_data, lambdas, k, 100)

    println()
    #= println("Difference between ADMM results: ", vecnorm(B_admm - B_admm2)) =#
    println("ADMM, oracle lambda: ", lambda1_oracle, ", difference: ", err1)
    println("ADMM, cv lambda: ", lambda1_cv, ", difference: ", vecnorm(B1_cv - pop_data.B))

    println("LH, oracle lambda: ", lambda2_oracle, ", difference: ", err2)
    println("LH, cv lambda: ", lambda2_cv, ", difference: ", vecnorm(B2_cv - pop_data.B))
  end

  function test_emp_data()
    p = 5
    Js_ind = [[1, 2], [3, 4]]
    Xs = []
    push!(Xs, randn(p, 10))
    push!(Xs, randn(p, 20))

    global emp_data = EmpiricalData(Xs, Js_ind)
  end

  function combined_oracle_screen(admm_data,
                                  lh_data;
                                  ps = [20],
                                  ns = [10000],
                                  ds = [5],
                                  ks = [5],
                                  trials = 1,
                                  scales = [0.8],
                                  experiment_type = "binary",
                                  vert = 5,
                                  horz = 5,
                                  lambdas = logspace(-1, -4, 50),
                                  prefix = "debug2.bin",
                                  force_well_conditioned = false,
                                  bad_init = false,
                                  graph_type = "random",
                                  missing_exps = [0],
                                  B_fixed = [],
                                  constant_n = false,
                                  cv = false,
                                  constraints = logspace(1, -4, 50)
                                 )

    global combined_results = []

    debug_data = []

    if length(ARGS) >= 1
      suffix = ARGS[1]
    else
      suffix = ""
    end

    for n in ns,
      p in ps,
      d in ds,
      k in ks,
      scale in scales,
      missing in missing_exps

      errors1_trials = zeros(trials)
      errors2_trials = zeros(trials)
      errors1_bad_trials = zeros(trials)
      errors2_bad_trials = zeros(trials)
      errors_lh_trials = zeros(trials)
      errors_lh_bad_trials = zeros(trials)
      errors_noconstr_trials = zeros(trials)
      errors_noconstr_bad_trials = zeros(trials)
      errors_llc_trials = zeros(trials)
      lambdas1_trials = zeros(trials)
      lambdas2_trials = zeros(trials)
      lambdas_llc_trials = zeros(trials)
      lambdas_lh_trials = zeros(trials)
      lambdas1_bad_trials = zeros(trials)
      lambdas2_bad_trials = zeros(trials)
      lambdas_lh_trials = zeros(trials)
      lambdas_lh_bad_trials = zeros(trials)
      lambdas_noconstr_trials = zeros(trials)
      lambdas_noconstr_bad_trials = zeros(trials)
      ground_truth_norms = zeros(trials)
      constraints_trials = zeros(trials)

      conditioning = []
      statuses1 = []
      statuses1_bad = []

      weak_cons = falses(trials)
      strong_cons = falses(trials)
      cyclics = falses(trials)

      for trial in 1:trials
        tic()
        println("Generating data")
        #= combined = loadvar(fname)[1] =#
        #= pop_data = combined[1] =#
        #= emp_data = combined[2] =#
        #= global pop_data = PopulationData(p, d, scale, experiment_type, graph_type = "overlap_cycles", k = k, horz = horz, vert = vert) =#
        #= global pop_data = PopulationData(p, d, scale, experiment_type, graph_type = "clusters") =#
        #= global pop_data = PopulationData(p, d, scale, experiment_type, graph_type = "random_no_norm") =#
        global pop_data

        if force_well_conditioned
          ill_conditioned = true
          ind_conditioning = zeros(0)
          while ill_conditioned
            pop_data = PopulationData(p, d, scale, experiment_type, graph_type = graph_type, missing_exps = missing, B_fixed = B_fixed, horz = horz, vert = vert, k = k)
            ind_conditioning = cond.(pop_data.thetas)
            if maximum(ind_conditioning) < 1e3
              ill_conditioned = false
            end
          end
        else
          pop_data = PopulationData(p, d, scale, experiment_type, graph_type = graph_type, missing_exps = missing, B_fixed = B_fixed, horz = horz, vert = vert, k = k)
          ind_conditioning = cond.(pop_data.thetas)
        end

        #= println("Condition number: ", cond(eye(p) - pop_data.B)) =#

        push!(conditioning, ind_conditioning)

        # Determine connectedness
        G = DiGraph(pop_data.B)
        strong_cons[trial] = is_strongly_connected(G)
        weak_cons[trial] = is_weakly_connected(G)
        cyclics[trial] = is_cyclic(G)

        # Generate samples
        n_effective = constant_n ? ceil(Int, n/pop_data.E) : n
        store_samples = cv
        emp_data = EmpiricalData(pop_data, n_effective, store_samples = store_samples)
        #= push!(debug_data, [pop_data, emp_data]) =#
        #= savevar(fname, debug_data) =#
        ground_truth_norms[trial] = vecnorm(pop_data.B)

        # Reset parameters
        reinit_admm(admm_data, p, emp_data.E)
        reinit_quic_data(admm_data.quic_data, p)
        reinit_lh_data(lh_data, p)
        lh_data.use_constraint = true

        global errors1, errors2, B1, B2
        if cv
          # Cross validation run
          kfold = 3
          (B1, B2, lh1, lh2, lambda1, lambda2, constraint, lhs1, lhs2) = combined_cv(emp_data, admm_data, lh_data, lambdas, constraints, kfold)
          push!(constraints_trials, constraint)
          err1 = vecnorm(B1 - pop_data.B)
          err2 = vecnorm(B2 - pop_data.B)
        else
          # Comined run with continuation
          (B1, B2, err1, err2, lambda1, lambda2, errors1, errors2, status1) = combined_oracle(pop_data, emp_data, admm_data, lh_data, lambdas)
          push!(statuses1, status1)
          # Run without constraint
          lh_data.use_constraint = false
          (B_noconstr, err_noconstr, lambda_noconstr, errors_noconstr) = min_constr_lh_oracle(pop_data, emp_data, lh_data, lambdas) 

          # Same thing, with bad initialization
          if bad_init
            B0_bad = 10 * triu(randn(p,p), 1)
            admm_data.B0 = B0_bad
            admm_data.duals = [zeros(emp_data.p, emp_data.p) for e = 1:emp_data.E] # duals
            admm_data.quic_data.inner_mult = 1
            lh_data.use_constraint = true
            (B1_bad, B2_bad, err1_bad, err2_bad, lambda1_bad, lambda2_bad, errors1_bad, errors2_bad, status1_bad) = combined_oracle(pop_data, emp_data, admm_data, lh_data, lambdas)
            push!(statuses1_bad, status1_bad)
            lh_data.use_constraint = false
            lh_data.B0 = B0_bad
            (B_noconstr_bad, err_noconstr_bad, lambda_noconstr_bad, errors_noconstr_bad) = min_constr_lh_oracle(pop_data, emp_data, lh_data, lambdas) 
          end

          # Run only likelihood
          println("Run likelihood only")
          lh_data.continuation = false
          lh_data.use_constraint = false
          lh_data.B0 = zeros(p, p)
          (B_lh, err_lh, lambda_lh, errors_lh) = min_constr_lh_oracle(pop_data, emp_data, lh_data, lambdas) 

          if bad_init
            lh_data.B0 = B0_bad
            (B_lh_bad, err_lh_bad, lambda_lh_bad, errors_lh_bad) = min_constr_lh_oracle(pop_data, emp_data, lh_data, lambdas) 
          end

          # Save variables over trials
          errors_noconstr_trials[trial] = err_noconstr
          errors_lh_trials[trial] = err_lh


          lambdas_noconstr_trials[trial] = lambda_noconstr
          lambdas_lh_trials[trial] = lambda_lh

          if bad_init
            errors1_bad_trials[trial] = err1_bad
            errors2_bad_trials[trial] = err2_bad
            errors_noconstr_bad_trials[trial] = err_noconstr_bad
            errors_lh_bad_trials[trial] = err_lh_bad

            lambdas1_bad_trials[trial] = lambda1_bad
            lambdas2_bad_trials[trial] = lambda2_bad
            lambdas_noconstr_bad_trials[trial] = lambda_noconstr_bad
            lambdas_lh_bad_trials[trial] = lambda_lh_bad
          end
        end

        # Run LLC
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

        push!(combined_results, Dict("n"=>n, "p"=>p, "d"=>d,
                                     "k"=>k,
                                     "std"=>scale,
                                     "missing_exps"=>missing,
                                     "conditioning"=>conditioning,
                                     "statuses1"=>statuses1,
                                     "statuses1_bad"=>statuses1_bad,
                                     "errs1"=> errors1_trials,
                                     "errs2"=>errors2_trials,
                                     "errs_noconstr" => errors_noconstr_trials,
                                     "errs1_bad"=> errors1_bad_trials,
                                     "errs2_bad"=>errors2_bad_trials,
                                     "errs_noconstr_bad" => errors_noconstr_bad_trials,
                                     "errs_lh" => errors_lh_trials,
                                     "errs_lh_bad" => errors_lh_bad_trials,
                                     "errs_llc"=>errors_llc_trials,
                                     "lambdas1"=>lambdas1_trials,
                                     "lambdas2"=>lambdas2_trials,
                                     "lambdas1_bad"=> lambdas1_bad_trials,
                                     "lambdas2_bad"=>lambdas2_bad_trials,
                                     "lambdas_noconstr_bad" => lambdas_noconstr_bad_trials,
                                     "lambdas_noconstr"=>lambdas2_trials,
                                     "lambdas_llc"=>lambdas_llc_trials,
                                     "lambdas_lh"=>lambdas_lh_trials,
                                     "lambdas_lh_bad"=>lambdas_lh_bad_trials,
                                     "weak_cons" => weak_cons,
                                     "strong_cons" => strong_cons,
                                     "cyclics" => cyclics,
                                     "exps" => pop_data.E,
                                     "gt"=>ground_truth_norms,
                                     "constant_n"=>constant_n,
                                     "constraints"=>constraints_trials
                                    ))

        mkpath("results_" * prefix)
        fname = joinpath("results_" * prefix, "results_" * suffix * ".bin")
        open(fname, "w") do file
          serialize(file, combined_results)
        end

        toc()
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

  function gen_test()
    p = 100
    d = 5
    matrix_std = 0.8
    experiment_type = "bounded"
    global pop_data1 = PopulationData(p, d, matrix_std, experiment_type, k = 4)
    global pop_data2 = PopulationData(p, d, matrix_std, experiment_type, k = 2, graph_type = "clusters")
  end

  function k_fold_test()
    p = 100
    d = 5
    n = 1000
    matrix_std = 0.8
    experiment_type = "binary"
    global pop_data = PopulationData(p, d, matrix_std, experiment_type)
    global emp_data = EmpiricalData(pop_data, n, store_samples = true)
    global emp_data_test, emp_data_train
    (emp_data_train, emp_data_test) = k_fold_split(pop_data, emp_data, 6, 6)
  end

  function admm_cv_test()
    experiment_type = "binary"
    n = 2000
    p = 50
    d = 5
    k = 3
    lambdas = flipdim(logspace(-4, -1, 30), 1)

    println("Generating data")
    global pop_data = PopulationData(p, d, matrix_std, experiment_type)

    emp_data = EmpiricalData(pop_data, n)

    B0 = zeros(p, p)
    constr_data = ConstraintData(p)
    admm_data = ADMMData(emp_data, constr_data, 1.0)
    admm_data.tol_abs = 5e-4
    admm_data.tol_rel = 1e-3
    admm_data.quic_data.print_stats = false
    admm_data.quic_data.tol = 1e-6
    admm_data.dual_balancing = true
    admm_data.bb = false
    admm_data.tighten = true
    admm_data.mu = 500
    admm_data.tau = 1.5
    admm_data.B0 = B0
    rho = 1.0
    #= rho = 0.8670764957916309 =#
    admm_data.rho = rho
    admm_data.low_rank = experiment_type != "binary"
    lh_data = VanillaLHData(p, 1, B0)
    lh_data.low_rank = experiment_type != "binary"
    lh_data.final_tol = 1e-3
    lh_data.use_constraint = true

    global errors, lhs

    global lh_val2 = lh(emp_data, lh_data, mat2vec(pop_data.B, emp_data), [])
    println("B lh: ", lh_val2)

    (B_cv, lh_val, lambda_cv, lhs) = min_admm_cv(emp_data, admm_data, lambdas, k)
    (B_oracle, err_oracle, lambda_oracle, errors) = min_admm_oracle(pop_data, emp_data, admm_data, lambdas)

    println("CV: ", lambda_cv, ", ", vecnorm(B_cv - pop_data.B))
    println("Oracle: ", lambda_oracle, ", ", vecnorm(B_oracle - pop_data.B))

  end

  function lh_cv_test()
    experiment_type = "binary"
    n = 3000
    p = 20
    d = 4
    k = 2
    lambdas = flipdim(logspace(-4, 0, 10), 1)
    constraints = flipdim(logspace(-2, 2, 20), 1)

    println("Generating data")
    global pop_data = PopulationData(p, d, matrix_std, experiment_type)

    emp_data = EmpiricalData(pop_data, n)

    B0 = zeros(p, p)
    #= B0 = 0.01 * randn(p, p) =#
    constr_data = ConstraintData(p)
    lh_data = VanillaLHData(p, 1, B0)
    lh_data.low_rank = experiment_type != "binary"
    lh_data.final_tol = 1e-3
		lh_data.x_base = mat2vec(pop_data.B, emp_data)
    lh_data.upper_bound = vecnorm(pop_data.B)^2
    lh_data.use_constraint = true
    lh_data.continuation = true

    global errors, lhs
    global B_cv, B_oracle, B_cv2, B_oracle2

    println(pop_data.B)
    global lh_val2 = lh(emp_data, lh_data, mat2vec(pop_data.B, emp_data), [])
    println("B lh: ", lh_val2)

    #= (B_cv, lh_val, lambda_cv, ub_cv, lhs) = min_constr_lh_cv(pop_data, emp_data, lh_data, lambdas, k, constraints) =#
    println(lh_data.B0)
    (B_cv, lh_val, lambda_cv, ub_cv, lhs) = min_constr_lh_cv(emp_data, lh_data, lambdas, k)
    lh_data.lambda = lambda_cv
    B_cv2 = copy(min_constraint_lh(emp_data, lh_data))
    println(lh_data.B0)
    (B_oracle, err_oracle, lambda_oracle, errors) = min_constr_lh_oracle(pop_data, emp_data, lh_data, lambdas)
    lh_data.lambda = lambda_oracle
    B_oracle2 = copy(min_constraint_lh(emp_data, lh_data))

    println("CV: lambda = ", lambda_cv, ", upper bound = ", ub_cv, ", distance = ", vecnorm(B_cv - pop_data.B))
    println("Oracle: lambda = ", lambda_oracle, ", upper bound = ", vecnorm(pop_data.B)^2, ", distance = ", vecnorm(B_oracle - pop_data.B))
    println("Diff cv: ", vecnorm(B_cv - B_cv2))
    println("Diff oracle: ", vecnorm(B_oracle - B_oracle2))
  end

  using Plots
  function test_condition_number()
    plotlyjs()

    p = 200
    d = 5
    std = 0.8
    repeats = 100
    n = 1000
    I = eye(p)
    global conds = zeros(repeats)
    global conds_emp = zeros(repeats)
    global pop_data

    for i = 1:repeats
      println("Repeat: ", i)
      #= B = gen_b(p, d, std/d) =#
      pop_data = PopulationData(p, d, std, "binary")
      emp_data = EmpiricalData(pop_data, n)
      #= conds[i] = cond(I - B) =#
      conds[i] = maximum(cond.(pop_data.thetas))
      #= conds_emp[i] = maximum(cond.(emp_data.sigmas_emp)) =#
    end
    Plots.histogram(conds)
  end

	#= fast_lh_test() =#
	#= lbfgsb_test_2() =#
	#= quic_test() =#
  #= graph_lasso_admm_test() =#
  #= admm_test() =#
  #= admm_oracle_test() =#
  #= llc_test() =#
  #= lh_oracle_test() =#
  #= combined_oracle_test() =#
  #= combined_llc_screen() =#
	#= l2_constr_test() =#
	#= l2_constr_test_sticky() =#
  #= gen_test() =#
  #= k_fold_test() =#
  #= admm_cv_test() =#
  #= lh_cv_test() =#
  #= test_condition_number() =#


  tic()
  #= combined_oracle_screen() =#

  # Choose task
  if length(ARGS) >= 2
    task = ARGS[2]
  else
    task = ""
  end

  # Set parameters
  p = 10
  B0 = zeros(p, p)
  constr_data = ConstraintData(p)
  admm_data = ADMMData(p, 1, constr_data, 1.0)
  admm_data.tol_abs = 5e-4
  admm_data.tol_rel = 1e-3
  admm_data.quic_data.print_stats = false
  admm_data.quic_data.tol = 1e-5
  admm_data.quic_data.tol_rel = 1e-4
  admm_data.quic_data.inner_tol = 1e-6
  admm_data.quic_data.inner_max_iterations = 2000
  admm_data.quic_data.max_iterations = 1000
  admm_data.quic_data.dual_balancing =true
  admm_data.graph_lasso_method = "quic"
  admm_data.dual_balancing = true
  admm_data.bb = false
  admm_data.tighten = false
  admm_data.mu = 4
  admm_data.tau = 1.5
  admm_data.B0 = B0
  rho = 1.0
  #= rho = 0.8670764957916309 =#
  admm_data.rho = rho
  admm_data.low_rank = experiment_type != "binary"
  #= admm_data.low_rank = true =#

  lh_data = VanillaLHData(p, 1, B0)
  lh_data.low_rank = experiment_type != "binary"
  #= lh_data.low_rank = true =#
  lh_data.final_tol = 1e-3
  lh_data.use_constraint = true

  # Clusters
  constr_data = ConstraintData(p)
  admm_data = ADMMData(p, 1, constr_data, 1.0)
  admm_data.tol_abs = 5e-4
  admm_data.tol_rel = 1e-3
  admm_data.quic_data.print_stats = false
  admm_data.quic_data.tol = 1e-6
  admm_data.quic_data.tol_rel = 1e-4
  admm_data.quic_data.inner_tol = 1e-7
  admm_data.quic_data.inner_max_iterations = 2000
  admm_data.quic_data.max_iterations = 1000
  admm_data.quic_data.dual_balancing = true
  admm_data.graph_lasso_method = "quic"
  admm_data.dual_balancing = true
  admm_data.bb = false
  admm_data.tighten = false
  admm_data.mu = 4
  admm_data.tau = 1.5
  admm_data.B0 = B0
  admm_data.warm_start = false
  admm_data.max_iterations = 200
  rho = 1.0
  #= rho = 0.8670764957916309 =#
  admm_data.rho = rho
  admm_data.low_rank = experiment_type != "binary"
  #= admm_data.low_rank = true =#

  lh_data = VanillaLHData(p, 1, B0)
  lh_data.low_rank = experiment_type != "binary"
  #= lh_data.low_rank = true =#
  lh_data.final_tol = 1e-3
  lh_data.use_constraint = true

  if task == "clusters_varn"
    # Clusters, varn
    combined_oracle_screen(
                           admm_data,
                           lh_data,
                           ps = [32],
                           ns = map(x -> ceil(Int32, x), logspace(log10(20), log10(20000), 12)),
                           #= ns = 2000, =#
                           ds = [3],
                           trials = 1,
                           scales = [0.8],
                           experiment_type = "binary",
                           force_well_conditioned = false,
                           prefix = "clusters_varn2",
                           lambdas = flipdim(logspace(-4, 1, 50), 1),
                           graph_type = "clusters"
                          )

  elseif task == "clusters_vard"
    # Clusters, vard
    combined_oracle_screen(
                           admm_data,
                           lh_data,
                           ps = [32],
                           #= ns = map(x -> ceil(Int32, x), logspace(log10(20), log10(20000), 12)), =#
                           ns = 2000,
                           ds = 1:10,
                           trials = 1,
                           scales = [0.8],
                           experiment_type = "binary",
                           force_well_conditioned = false,
                           prefix = "clusters_vard",
                           lambdas = flipdim(logspace(-4, 0, 50), 1),
                           graph_type = "clusters"
                          )

  elseif task == "clusters_vard_norm"
    # Clusters, vard_norm
    combined_oracle_screen(
                           admm_data,
                           lh_data,
                           ps = [32],
                           #= ns = map(x -> ceil(Int32, x), logspace(log10(20), log10(20000), 12)), =#
                           ns = 2000,
                           ds = 1:10,
                           trials = 1,
                           scales = [0.8/sqrt(10)],
                           experiment_type = "binary",
                           force_well_conditioned = false,
                           prefix = "clusters_vard_norm",
                           lambdas = flipdim(logspace(-4, 1, 50), 1),
                           graph_type = "clusters_norm"
                          )

  elseif task == "clusters_varp"
    # Clusters, varp
    admm_data.early_stop = false
    combined_oracle_screen(
                           admm_data,
                           lh_data,
                           ps = 10:5:60,
                           #= ns = map(x -> ceil(Int32, x), logspace(log10(20), log10(20000), 12)), =#
                           #= vert = 5, =#
                           #= horz = 5, =#
                           ns = ceil(Int, 2 * log2(30) * 2000),
                           ds = [4],
                           trials = 1,
                           scales = [0.8],
                           experiment_type = "binary",
                           force_well_conditioned = false,
                           prefix = "clusters_varp2",
                           lambdas = flipdim(logspace(-4, 1, 50), 1),
                           graph_type = "clusters",
                           constant_n = true
                          )
    admm_data.early_stop = true

  elseif task == "clusters_missing"
    # Random, missing exps
    admm_data.low_rank = true
    lh_data.low_rank = true

    combined_oracle_screen(
                           admm_data,
                           lh_data,
                           ps = [32],
                           ns = 2000 * 32,
                           ds = [3],
                           trials = 1,
                           scales = [0.8],
                           experiment_type = "single",
                           force_well_conditioned = false,
                           prefix = "clusters_missing",
                           lambdas = flipdim(logspace(-4, 1, 50), 1),
                           graph_type = "clusters",
                           missing_exps = 0:31,
                           constant_n = true
                          )

    admm_data.low_rank = false
    lh_data.low_rank = false

  elseif task == "cycles_varp"
    # Cycles, varp
    combined_oracle_screen(
                           admm_data,
                           lh_data,
                           ps = [20, 34, 48, 62],
                           #= ns = map(x -> ceil(Int32, x), logspace(log10(20), log10(20000), 12)), =#
                           vert = 5,
                           horz = 5,
                           ns = ceil(Int, 2 * log2(62) * 2000),
                           ds = [3],
                           trials = 1,
                           scales = [0.8],
                           experiment_type = "binary",
                           force_well_conditioned = false,
                           prefix = "cycles_varp",
                           lambdas = flipdim(logspace(-4, 1, 50), 1),
                           graph_type = "overlap_cycles",
                           constant_n = true
                          )

  elseif task == "cycles_varn"
    # Cycles, varn
    combined_oracle_screen(
                           admm_data,
                           lh_data,
                           ps = [34],
                           ns = map(x -> ceil(Int32, x), logspace(log10(20), log10(20000), 12)),
                           vert = 5,
                           horz = 5,
                           ds = [3],
                           trials = 1,
                           scales = [0.8],
                           experiment_type = "binary",
                           force_well_conditioned = false,
                           prefix = "cycles_varn",
                           lambdas = flipdim(logspace(-4, 1, 50), 1),
                           graph_type = "overlap_cycles"
                          )

  elseif task == "cycles_missing"
    # Random, missing exps
    admm_data.low_rank = true
    lh_data.low_rank = true

    combined_oracle_screen(
                           admm_data,
                           lh_data,
                           ps = [34],
                           vert = 5,
                           horz = 5,
                           ns = 2000 * 34,
                           ds = [3],
                           trials = 1,
                           scales = [0.8],
                           experiment_type = "single",
                           force_well_conditioned = false,
                           prefix = "cycles_missing",
                           lambdas = flipdim(logspace(-4, 1, 50), 1),
                           graph_type = "clusters",
                           missing_exps = 0:33,
                           constant_n = true
                          )

    admm_data.low_rank = false
    lh_data.low_rank = false

  elseif task == "rand_missing"
    # Random, missing exps
    admm_data.low_rank = true
    lh_data.low_rank = true

    combined_oracle_screen(
                           admm_data,
                           lh_data,
                           ps = [30],
                           ns = 2000 * 30,
                           ds = [5],
                           trials = 1,
                           scales = [0.8],
                           experiment_type = "single",
                           force_well_conditioned = false,
                           prefix = "rand_missing",
                           lambdas = flipdim(logspace(-4, 1, 50), 1),
                           graph_type = "random",
                           missing_exps = 0:29,
                           constant_n = true
                          )

    admm_data.low_rank = false
    lh_data.low_rank = false

  elseif task == "rand_varn"
    # Random, varn
    combined_oracle_screen(
                           admm_data,
                           lh_data,
                           ps = [30],
                           #= ns = 2000, =#
                           ns = map(x -> ceil(Int32, x), logspace(log10(20), log10(20000), 12)),
                           ds = [4],
                           trials = 1,
                           scales = [0.8],
                           experiment_type = "binary",
                           force_well_conditioned = false,
                           prefix = "rand_varn",
                           lambdas = flipdim(logspace(-4, 1, 50), 1),
                           graph_type = "random"
                          )

  elseif task == "rand_varp"
    # Random, varp
    admm_data.early_stop = false
    combined_oracle_screen(
                           admm_data,
                           lh_data,
                           ps = 10:5:30,
                           #= ns = map(x -> ceil(Int32, x), logspace(log10(20), log10(20000), 12)), =#
                           #= vert = 5, =#
                           #= horz = 5, =#
                           ns = ceil(Int, 2 * log2(30) * 2000),
                           ds = [4],
                           trials = 1,
                           scales = [0.8],
                           experiment_type = "binary",
                           force_well_conditioned = false,
                           prefix = "rand_varp_noearly",
                           lambdas = flipdim(logspace(-4, 1, 50), 1),
                           graph_type = "random",
                           constant_n = true
                          )
    admm_data.early_stop = true

  elseif task == "rand_vard_norm"
    # Random, vard
    combined_oracle_screen(
                           admm_data,
                           lh_data,
                           ps = [60],
                           ns = 2000,
                           #= ns = map(x -> ceil(Int32, x), logspace(log10(20), log10(20000), 12)), =#
                           ds = 1:10,
                           trials = 1,
                           scales = [0.8/sqrt(10)],
                           experiment_type = "binary",
                           force_well_conditioned = false,
                           prefix = "rand_vard_norm",
                           lambdas = flipdim(logspace(-4, 1, 50), 1),
                           graph_type = "random_norm"
                          )

  elseif task == "rand_vark"
    # Random, vark
    combined_oracle_screen(
                           admm_data,
                           lh_data,
                           ps = [30],
                           ns = [2000 * 30],
                           #= ns = map(x -> ceil(Int32, x), logspace(log10(20), log10(20000), 12)), =#
                           ds = [3],
                           ks = 1:10,
                           trials = 1,
                           scales = [0.8/sqrt(3)],
                           experiment_type = "bounded",
                           force_well_conditioned = false,
                           prefix = "rand_vare_norm",
                           lambdas = flipdim(logspace(-4, 1, 50), 1),
                           graph_type = "random_norm",
                           constant_n = true
                          )

  elseif startswith(task, "semi_synth")

  #=   # Cai paper =#
    B = zeros(39, 39)
    B[24,4] = -1.6206
    B[24,5] = 0.1257
    B[29,5] = 0.1925
    B[24,22] = 0.1008
    B[28,22] = 0.2940
    B[29,22] = 0.2988
    B[22,24] = 0.7366
    B[29,28] = -0.0406
    B[5,29] = 0.3378
    B[22,29] = 0.8308
    B[28,29] = -0.6262
    B[22,32] = -0.0882
    B[28,32] = 0.2409

    d = maximum(sum(B .!= 0, 2))
    (p, _) = size(B)

    if task == "semi_synth_varn"
      combined_oracle_screen(
                             admm_data,
                             lh_data,
                             ps = [p],
                             ds = [d],
                             ns = map(x -> ceil(Int32, x), logspace(log10(20), log10(20000), 12)),
                             trials = 1,
                             experiment_type = "binary",
                             force_well_conditioned = false,
                             prefix = "semi_synth",
                             lambdas = flipdim(logspace(-4, 1, 50), 1),
                             graph_type = "given",
                             B_fixed = copy(B)
                            )

    elseif task == "semi_synth_vare"
    # Varying E
      admm_data.low_rank = true
      lh_data.low_rank = true

      combined_oracle_screen(
                             admm_data,
                             lh_data,
                             ps = [p],
                             ds = [d],
                             missing_exps = 0:38,
                             ns = 2000 * 39,
                             trials = 1,
                             experiment_type = "single",
                             force_well_conditioned = false,
                             prefix = "semi_synth_missing",
                             lambdas = flipdim(logspace(-4, 1, 50), 1),
                             graph_type = "given",
                             B_fixed = copy(B),
                             constant_n = true
                            )

      admm_data.low_rank = false
      lh_data.low_rank = false
    end

  elseif task == "rand_cv_varn"
    println("Rand_cv_varn")
    # Random, varn
    combined_oracle_screen(
                           admm_data,
                           lh_data,
                           ps = [20],
                           #= ns = 2000, =#
                           ns = map(x -> ceil(Int32, x), logspace(log10(100), log10(20000), 12)),
                           ds = [3],
                           trials = 1,
                           scales = [0.8],
                           experiment_type = "binary",
                           force_well_conditioned = false,
                           prefix = "rand_cv_varn",
                           lambdas = flipdim(logspace(-4, 1, 50), 1),
                           graph_type = "random",
                           cv = true
                          )

  end

  toc()

end
