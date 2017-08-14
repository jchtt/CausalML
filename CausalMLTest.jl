# vim: ts=2 sw=2 et

module CausalMLTest
	#= if isdefined(:CausalML) =#
	#= 	reload("CausalML") =#
	#= end =#
	import CausalML
	reload("CausalML")
	using CausalML

	using Lbfgsb

  # Optional
	using Calculus
  using Convex
  using SCS

	const experiment_type = "single"
	const p = 10
	const d = 2
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
		lh_data = VanillaLHData(p, lambda, B0)
		lh_data.x_base = mat2vec(pop_data.B, emp_data)
		lh_data.upper_bound = 0.0001
		@time global B_lbfgsb = min_vanilla_lh(emp_data, lh_data, low_rank = true)
		global s
		global B_constr
		@time (B_constr, s) = min_constraint_lh(emp_data, lh_data)
		println("LBFGSB, difference: ", vecnorm(B_lbfgsb - pop_data.B))
		println("LBFGSB, constraint: ", vecnorm(B_constr - pop_data.B))
		println("Difference: ", vecnorm(B_constr - B_lbfgsb))
		println("Slack difference: ", s - vecnorm(B_constr-pop_data.B)^2)
		println("Slack: ", s)
	end

	function quic_test()
    data = QuadraticPenaltyData(p)
    lambda = 1e-3
    rho = 0.0
		@time global theta1 = quic2(p, emp_data, emp_data.sigmas_emp[1], rho = rho, lambda = lambda, print_stats = false)
		@time global theta2 = quic3(emp_data, data, emp_data.sigmas_emp[1], rho = rho, lambda = lambda, print_stats = false)
    Profile.clear()
		@time global theta1 = quic2(p, emp_data, emp_data.sigmas_emp[1], rho = rho, lambda = lambda)
		@time global theta2 = quic3(emp_data, data, emp_data.sigmas_emp[1], rho = rho, lambda = lambda)
		println("Difference: ", vecnorm(theta1 - pop_data.thetas[1])^2)
    println("Inverse emp cov difference: ", vecnorm(inv(emp_data.sigmas_emp[1]) - pop_data.thetas[1])^2)
    println("Difference from each other: ", vecnorm(theta1 - theta2))
    
    xvar = Variable(p, p)
    sigma = emp_data.sigmas_emp[1]
    theta_prime = eye(p)
    problem = minimize(sum(sigma .* xvar) - logdet(xvar) + lambda * vecnorm(xvar, 1) + rho/2 * vecnorm(xvar - theta_prime)^2)
    solve!(problem, SCSSolver())

    x_convex = xvar.value
    #= println(vecnorm(x_convex - x_quic)) =#
    #= println(vecnorm(thetas[1] - x_convex)) =#
    #= println(vecnorm(thetas[1] - x_quic)) =#
    #= println(vecnorm(x_quic2 - x_fb)) =#
    println("Difference to convex program: ", vecnorm(theta2 - x_convex))
    #= println(vecnorm(x_convex - x_fb)) =#
	end

	function admm_test()
		B0 = zeros(p, p)
    #= constr_data = ConstraintData(p) =#
    #= admm_data = ADMMData(emp_data, qu_data, constr_data) =#
    #= rho = 1.0 =#
    #= lambda = 1e-2 =#
    #= @time global B_admm = min_admm(emp_data, admm_data, lambda, B0, rho) =#
    #= println("ADMM, difference: ", vecnorm(B_admm - pop_data.B)) =#

    constr_data = ConstraintData2(p)
    lambda = 1e-2
    admm_data = ADMMData2(emp_data, constr_data, lambda)
    rho = 1.0
    admm_data.rho = rho
    @time global B_admm2 = min_admm2(emp_data, admm_data, lambda, B0)
    println()
    #= println("Difference between ADMM results: ", vecnorm(B_admm - B_admm2)) =#
    println("ADMM, difference: ", vecnorm(B_admm2- pop_data.B))
	end

	#= fast_lh_test() =#
	#= lbfgsb_test_2() =#
	#= l2_constr_test() =#
	#= quic_test() =#
  admm_test()
end
