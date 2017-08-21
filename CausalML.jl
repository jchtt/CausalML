# vim: ts=2 sw=2 et
module CausalML
export gen_b, PopulationData, EmpiricalData, lh, l1_wrap, mat2vec, vec2mat, min_vanilla_lh, VanillaLHData, min_constraint_lh, quic_old, quic, QuadraticPenaltyData, min_admm, ConstraintData, ADMMData, QuadraticPenaltyData_old, min_admm_old, ConstraintData_old, ADMMData_old, min_admm_oracle, llc, symmetrize!, min_constr_lh_oracle, combined_oracle, savevar

#using Plots
using Lbfgsb
using NLopt
#= using FiniteDiff =#
#= using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile =#
#using ForwardDiff
using BenchmarkTools
using StatsFuns
using Base.LinAlg.LAPACK
using Calculus
#= using RGlasso =#
#= using JLD =#
using Lasso

first_pass = true
admm_path = []
debugfname = "debug.bin"

#= include("Tools.jl") =#
#= using .Tools =#

#= include("RQuic.jl") =#
#= using .RQuic =#

#= include("Liblbfgs.jl/Liblbfgs.jl") =#
#= if ~("./Liblbfgs.jl" in LOAD_PATH) =#
#= 	push!(LOAD_PATH, "./Liblbfgs.jl") =#
#= end =#
#= using Liblbfgs =#

function savevar(fname, var)
  open(fname, "w") do file
    serialize(file, var)
  end
end

function symmetrize!(A)
  n = LinAlg.checksquare(A)
  @inbounds for i = 1:n, j = i:n
    A[i,j] = (B[i,j] + B[j,i]) / 2
  end
  LinAlg.copytri!(A, 'U')
end

# Generate a random matrix B
function gen_b(p, d, std)
 B = zeros(p, p)
 for i = 1:p
  perm = randperm(p - 1)
  perm[perm .>= i] += 1
  #= B[i, perm[1:d]] = std*randn(d) =#
  B[i, perm[1:d]] = std*(2*rand(d)-1)
 end
 return B
end

function hard_thresh(x, t)
  return abs(x) >= t ? x : 0.0
end

type PopulationData
  p::Int64
  d::Int64
  std::Float64
  B::Array{Float64,2}
  sigmas
  thetas
  Ainvs
  Us
  Us_ind
  Js_ind
  I
  E

  function PopulationData(p, d, std, experiment_type)
    ret = new()
    ret.p = p
    ret.d = d
    ret.std = std
    #= ret.B = gen_b(p, d, std/sqrt(d*p)) =#
    ret.B = gen_b(p, d, std/d)
    #= ret.B = hard_thresh.(ret.B, std/d) =#

    # Generate experiment data
    ret.Us = []
    ret.Us_ind = []
    ret.Js_ind = []
    ret.I = eye(p)
    ret.Ainvs = []
    ret.sigmas = []
    ret.thetas = []

    if experiment_type == "binary"
      for e = 0:log2(p)
        mask = mod(floor((0:(p-1))/2^e), 2)
        mask = mask .== 1
        push!(ret.Us_ind, find(~mask), find(mask))
        push!(ret.Js_ind, find(mask), find(~mask))
        # println(~(mask.==1))

        U = ones(p)
        U[mask] = 0
        U = diagm(U)

        U2 = ones(p)
        U2[~mask] = 0
        U2 = diagm(U2)

        push!(ret.Us, U, U2)
      end
    elseif experiment_type == "single"
      for e = 1:p
        mask = falses(p)
        mask[e] = true
        push!(ret.Us_ind, find(~mask))
        push!(ret.Js_ind, find(mask))
        # println(~(mask.==1))

        U = ones(p)
        U[mask] = 0
        U = diagm(U)

        push!(ret.Us, U)
      end
    end
    ret.E = length(ret.Us)

    # Generate concentration and covariance matrices
    for e = 1:ret.E
      # Population level matrices
      Ainv = I - ret.Us[e]*ret.B
      push!(ret.Ainvs, Ainv)
      theta = Ainv'*Ainv
      sigma = inv(theta)
      push!(ret.sigmas, sigma)
      push!(ret.thetas, theta)
    end

    return ret
  end
end

type EmpiricalData
  p::Int64
  n::Int64
  E
  I
  diag_idx
  offdiag_idx
  find_offdiag_idx
  dimsym
  symdiag_idx
  strict_lower_idx
  Us
  Us_ind
  Js_ind
  experiment_type
  sigmas_emp
  Xs

  function EmpiricalData(pop_data, n)
    ret = new()
    ret.p = pop_data.p
    ret.E = pop_data.E
    ret.n = n

    # Prepare fixed indices
    ret.diag_idx = diagm(trues(ret.p,1))
    ret.offdiag_idx = ~ret.diag_idx
    ret.find_offdiag_idx = find(~ret.diag_idx)
    ret.dimsym = div(ret.p*(ret.p+1), 2)

    ret.symdiag_idx = falses(ret.dimsym)
    cur = 1
    for i = 1:ret.p
      ret.symdiag_idx[cur+i-1] = true
      cur += i
    end

    ret.strict_lower_idx = tril(trues(ret.p,ret.p), -1)

    # Generate covariance matrices
    B = pop_data.B
    ret.Us = pop_data.Us
    ret.Us_ind = pop_data.Us_ind
    ret.Js_ind = pop_data.Js_ind
    ret.I = pop_data.I
    ret.sigmas_emp = []

    # Generate covariance matrices
    ret.Xs = []
    for e = 1:ret.E
      # Population level matrices
      Ainv = pop_data.Ainvs[e]

      # Empirical covariance
      Z = randn(ret.p, ret.n)
      #= push!(ret.Xs, Ainv \ Z) =#
      #= X = ret.Xs[e] =#
      X = Ainv \ Z
      sigma_emp = X*X'/n
      push!(ret.sigmas_emp, sigma_emp)
    end

    return ret
  end
end

# Index matrices for reuse
function initialize(p)
	#= global diag_idx, offdiag_idx, find_offdiag_idx, dimsym =#
	const diag_idx = diagm(trues(p, 1))
	const offdiag_idx = ~diag_idx
	const find_offdiag_idx = find(~diag_idx)
	const dimsym = div(p*(p+1), 2)
end

# Parameters
#= const experiment_type = "binary" =#
const experiment_type = "binary"
const p = 200
const d = 10
sigma = 0.8/d
lambda = 1e3
epsilon = 1e-5
n = Int32(1e4)
const dimsym = div(p*(p+1), 2)

symdiag_idx = falses(dimsym)
cur = 1
for i = 1:p
  symdiag_idx[cur+i-1] = true
  cur += i
end

const strict_lower_idx = tril(trues(p,p), -1)

# Conversion functions
function mat2vec(B, emp_data; reduced = true, symmetric = false, inplace = [])
  (p1, p2) = size(B)
  if symmetric
    dimsym = div(p*(p+1), 2)
    ret = zeros(dimsym)
    cur = 1
    for i = 1:p1
      ret[cur:cur + i - 1] = B[1:i, i]
      cur += i
    end
    return ret
  elseif reduced
    if length(inplace) > 0
      println("CALLED")
      for (I, J) in zip(eachindex(inplace), find_offdiag_idx)
        inplace[I] = B[J]
      end
    else
      return B[emp_data.offdiag_idx]
    end
  else
    return reshape(B, p1*p2)
  end
end

function vec2mat(b, emp_data; reduced = true, symmetric = false, inplace = [])
  p = length(b)
  sp = ceil(Int64, sqrt(p))
  if symmetric
    sp = ceil(Int64, sqrt(1/4 + 2*p) - 1/2)
    ret = [i <= j ? b[round(Int, j*(j-1)/2) + i] : b[round(Int, i*(i-1)/2) + j] for i = 1:sp, j = 1:sp]
    z = zero(eltype(b))
    #ret = [i <= j ? b[round(Int, j*(j-1)/2) + i] : z for i = 1:sp, j = 1:sp]
    #=
    ret = similar(reshape(b, sp, sp))
    cur = 1
    for i = 1:sp
      ret[1:i, i] = b[cur:cur + i - 1]
      cur += i
    end
    =#
    #ret += triu(ret, 1)'
    return ret
  elseif reduced
    if length(inplace) > 0
      ret = inplace
      fill!(ret, 0.0)
    else
      ret = zeros(sp, sp)
    end
    ret[emp_data.offdiag_idx] = b
    return ret
  else
    return reshape(b, sp, sp)
  end
end

function symmetrize!(A)
  A[:] += A'[:]
  A[:] ./= 2
end

function pd_inv(A)
  R = chol(A)
  Rinv = inv(R)
  ret = Rinv * Rinv'
  return ret
end

function pd_inv!(A, ret)
  ret[:] = copy(A)
  LAPACK.potrf!('L', ret)
  LAPACK.potri!('L', ret)
  tril!(ret)
  ret[:] += tril(ret, -1)'[:]
end

type VanillaLHData
  lambda # Regularization parameter
  B0 # Starting value
  B # TODO: Do I need this?
  Ainv::Array{Float64,2} # I - B, or I - U*B
  Ainv2::Array{Float64,2} # (I - B)' * (I - B)
  Ainv2e::Array{Float64,2} # (I - U*B)' * (I - U*B)
  Breds::Array{Array{Float64,2},1} # Low rank B updates
  Bredst::Array{Array{Float64,2},1} # Transposed low rank updates
  Bmuldiff::Array{Array{Float64,2},1} # Intermediate value in gradient calculation
  # Storage of Cholesky factorizations
  Cholfact::LinAlg.Cholesky
  CholfactFactor::Array{Float64,2}
  CholfactFactorT::Array{Float64,2}
  Cholfacte::LinAlg.Cholesky
  CholfacteFactor::Array{Float64,2}
  CholfacteFactorT::Array{Float64,2}
  # Vectors for lowrank Cholesky update
  update_vec::Array{Float64,1}
  downdate_vec::Array{Float64,1}
  diff::Array{Float64,2} # difference in gradient calculation
  diff_sum::Array{Float64,2} # running sum of differences
  Gmat::Array{Float64,2}
  Gmat_sum::Array{Float64,2} # Carry the sum of matrices in gradient calculation
  initialize_array::Bool # Should we initialize the multi-level arrays Breds, Bredst, Bmuldiff?
  # For assembling vector in l1 wrapper
  x_plus::Array{Float64,1} 
  x_minus::Array{Float64,1}
  x_total::Array{Float64,1}
  # For gradient storage in l1 wrapper
  grad_f::Array{Float64,1} 
  # For gradient of l1 wrapper
  ones::Array{Float64,1} 
  # Lower & upper bounds for augmented Lagrangian
  lb_aug::Array{Float64,1} 
  ub_aug::Array{Float64,1}
  # State variables for augmented Lagrangian
  x_old::Array{Float64,1} 
  x_new::Array{Float64,1} 
  final_tol::Float64 # tolerance for augmented Lagrangian method
  tau::Float64 # tolerance factor for dual rebalancing
  gamma::Float64 # multiplier for dual rebalancing
  aug_diff::Array{Float64,1} # Storage for difference vector
  x_base::Array{Float64,1} # Base vector for l2 constraint
  upper_bound::Float64 # Radius for l2 constraint
  low_rank::Bool # Use low rank decomposition
  s::Float64 # Starting value for constraint slack parameter
  dual::Float64 # Dual variable
end

function VanillaLHData(p, lambda, B0)
  return VanillaLHData(lambda, B0,
                       zeros(p, p), zeros(p, p),
                       zeros(p, p), zeros(p, p),
                       [], [], [],
                       LinAlg.Cholesky(zeros(p,p), :U),
                       zeros(p,p),
                       zeros(p,p),
                       LinAlg.Cholesky(zeros(p,p), :U),
                       zeros(p,p),
                       zeros(p,p),
                       zeros(p),
                       zeros(p),
                       zeros(p, p),
                       zeros(p, p),
                       zeros(p, p),
                       zeros(p, p),
                       true,
                       zeros(p*(p-1)),
                       zeros(p*(p-1)),
                       zeros(p*(p-1)),
                       zeros(p*(p-1)),
                       ones(p*(p-1)),
                       zeros(2*p*(p-1) + 1),
                       fill(Inf, 2*p*(p-1)+1),
                       zeros(2*p*(p-1)+1),
                       zeros(2*p*(p-1)+1),
                       1e-6, # final_tol
                       0.5, # gamma
                       1.5, # tau
                       zeros(p*(p-1)), # aug_diff
                       zeros(p*(p-1)),
                       Inf,
                       false, # low_rank
                       0.0, # s0
                       0.0, # dual
                      )
end

# Set up functions for lbfgsb
function lh(emp_data, data, b, g::Vector; reduced = true, low_rank = false)
  compute_grad = length(g) > 0

  # Convert to matrix
  B = vec2mat(b, emp_data, reduced = reduced)
  vec2mat(b, emp_data, inplace = data.B)
  I = emp_data.I
  p = emp_data.p

  if compute_grad
    fill!(g, 0.0)
    if low_rank
      fill!(data.Gmat_sum, 0.0)
      fill!(data.diff_sum, 0.0)
    end
  end

  val = 0.0

  fill!(data.Ainv, 0.0)
  #= Ainv = similar(B) =#
  #= theta_e = similar(B) =#
  #= delta_sigma = similar(B) =#
  #= grad_mat = similar(B) =#

  data.initialize_array = length(data.Breds) == 0

  if low_rank
    # Assemble (I - B)
    for i = 1:p
      data.Ainv[i,i] = 1.0
    end

    #= data.Ainv[:] .-= B[:] =#
    BLAS.axpy!(-1.0, data.B, data.Ainv)
    BLAS.gemm!('T', 'N', 1.0, data.Ainv, data.Ainv, 0.0, data.Ainv2)
    symmetrize!(data.Ainv2)

    copy!(data.CholfactFactor, data.Ainv2)
    # Use Symmetric() to handle round-off errors above
    cholfact!(Symmetric(data.CholfactFactor))
  end

  for e = 1:emp_data.E
    U = emp_data.Us[e]
    J_ind = emp_data.Js_ind[e]

    if low_rank
      if data.initialize_array
        push!(data.Breds, zeros(length(J_ind), p))
        push!(data.Bmuldiff, zeros(length(J_ind), p))
        push!(data.Bredst, zeros(p, length(J_ind)))
      end
      copy!(data.Breds[e], view(data.B, J_ind, :))
      transpose!(data.Bredst[e], data.Breds[e])
      copy!(data.Ainv2e, data.Ainv2)
      BLAS.axpy!(1.0, data.Breds[e], view(data.Ainv2e, J_ind, :))
      BLAS.axpy!(1.0, data.Bredst[e], view(data.Ainv2e, :, J_ind))
      BLAS.gemm!('N', 'N', -1.0, data.Bredst[e], data.Breds[e], 1.0, data.Ainv2e)
      symmetrize!(data.Ainv2e)

      val += BLAS.dot(p^2, data.Ainv2e, 1, emp_data.sigmas_emp[e], 1)

      copy!(data.CholfacteFactor, data.CholfactFactor)
      data.Cholfacte = LinAlg.Cholesky(data.CholfacteFactor, :U)

      for j = 1:length(J_ind)
        fill!(data.update_vec, 0.0)
        data.update_vec[J_ind[j]] = 1.0
        copy!(data.downdate_vec, view(data.Bredst[e], :, j))
        scale!(data.downdate_vec, -1.0)
        data.downdate_vec[J_ind[j]] += 1.0
        LinAlg.lowrankupdate!(data.Cholfacte, data.update_vec)
        LinAlg.lowrankdowndate!(data.Cholfacte, data.downdate_vec)
        #= copy!(data.diag, view(data.Cholfacte.factors, emp_data.diag_idx)) =#
      end
      try
        val -= sum(2*log(diag(data.Cholfacte.factors)))
      catch
        println("Warning: (I-U*B)'*(I-U*B) not positive definite.")
        val = Inf
      end
    else
      # Compute theta
      fill!(data.Ainv, 0.0)
      for i = 1:p
        data.Ainv[i, i] = 1
      end
      BLAS.axpy!(-1.0, view(data.B, emp_data.Us_ind[e], :), view(data.Ainv, emp_data.Us_ind[e], :))
      BLAS.gemm!('T', 'N', 1.0, data.Ainv, data.Ainv, 0.0, data.Ainv2)
      symmetrize!(data.Ainv2)
      val += BLAS.dot(p^2, emp_data.sigmas_emp[e], 1, data.Ainv2, 1)
      copy!(data.CholfactFactor, data.Ainv2)
      # Use Symmetric() to handle round-off errors above
      try
        cholfact!(Symmetric(data.CholfactFactor))
      catch y
        #= println("Cholesky exception!") =#
        #= println("lambda = ", data.lambda) =#
        #= println("x_base = ", data.x_base) =#
        #= println("B = ", data.B) =#
        #= println("Ainv = ", data.Ainv) =#
        #= println("Ainv2 = ", data.Ainv2) =#
        #= save("sticky_data3.jld", "emp_data", emp_data, "lh_data", data) =#
        #= throw(y) =#
        val = Inf
        return val
      end
      try
        val -= sum(2*log(diag(data.CholfactFactor)))
      catch
        println("Warning: (I-U*B)'*(I-U*B) not positive definite.")
        val = Inf
        return val
      end

      #= Ainv .= I - U*B =#
      #= theta_e .= Ainv' * Ainv =#
      #= val += sum(emp_data.sigmas_emp[e] .* theta_e) =#
      # println(val)
      #= try =#
      #=   val -= logdet(theta_e) =#
      #= catch =#
      #=   println("Warning: (I-U*B)'*(I-U*B) not positive definite.") =#
      #=   val = Inf =#
      #= end =#
      # println(val)
    end

    # Gradient calculation
    if compute_grad
      if low_rank
        fill!(data.Gmat, 0.0)
        copy!(data.diff, emp_data.sigmas_emp[e])
        LAPACK.potri!('U', data.CholfacteFactor)
        data.CholfacteFactor[emp_data.strict_lower_idx] = 0.0
        transpose!(data.CholfacteFactorT, data.CholfacteFactor)
        data.CholfacteFactor[emp_data.diag_idx] = 0.0
        BLAS.axpy!(1.0, data.CholfacteFactorT, data.CholfacteFactor)
        BLAS.axpy!(-1.0, data.CholfacteFactor, data.diff)

        BLAS.axpy!(1.0, data.diff, data.diff_sum)
        BLAS.axpy!(-2.0, view(data.diff, J_ind, :), view(data.Gmat_sum, J_ind, :))
        BLAS.gemm!('N', 'N', 2.0, data.Breds[e], data.diff, 0.0, data.Bmuldiff[e])
        BLAS.axpy!(1.0, data.Bmuldiff[e], view(data.Gmat_sum, J_ind, :))
      #= else =#
      #=   BLAS.gemm!('N', 'N', 2.0, data.Ainv, data.diff, 0.0, data.Gmat) =#
      #=   data.Gmat[J_ind,:] = 0.0 =#
      #=   BLAS.axpy!(1.0, view(data.Gmat, find_offdiag_idx), g) =#
      else
        copy!(data.diff, emp_data.sigmas_emp[e])
        LAPACK.potri!('U', data.CholfactFactor)
        data.CholfactFactor[emp_data.strict_lower_idx] = 0.0
        transpose!(data.CholfactFactorT, data.CholfactFactor)
        data.CholfactFactor[emp_data.diag_idx] = 0.0
        BLAS.axpy!(1.0, data.CholfactFactorT, data.CholfactFactor)
        BLAS.axpy!(-1.0, data.CholfactFactor, data.diff)
        BLAS.gemm!('N', 'N', 2.0, data.Ainv, data.diff, 0.0, data.Gmat)
        data.Gmat[emp_data.Js_ind[e],:] = 0.0
        BLAS.axpy!(-1.0, view(data.Gmat, emp_data.find_offdiag_idx), g)
        #= try =#
        #=   delta_sigma .= emp_data.sigmas_emp[e] - inv(theta_e) =#
        #= catch =#
        #=   println("Warning: (I-U*B)'*(I-U*B) not positive definite.") =#
        #=   delta_sigma .= emp_data.sigmas_emp[e] - pinv(theta_e) =#
        #= end =#
        #= grad_mat .= - 2 * U * Ainv * delta_sigma =#
        #= # println(grad_mat) =#
        #= g[:] += mat2vec(grad_mat, emp_data) =#
      end
    end
  end

  if compute_grad && low_rank
    BLAS.gemm!('N', 'N', 2.0, data.Ainv, data.diff_sum, 1.0, data.Gmat_sum)
    copy!(g, view(data.Gmat_sum, emp_data.find_offdiag_idx))
    scale!(-1.0, g)
  end

  data.initialize_array = false
  return val
end

function l1_wrap(x, g, lambda, f, data)
  dim = div(length(x),2)
  # x_plus = x[1:dim]
  copy!(data.x_plus, view(x, 1:dim))
  # x_minus = x[dim+1:2*dim]
  copy!(data.x_minus, view(x, (dim+1):2*dim))
  # x_total = x_plus - x_minus
  copy!(data.x_total, data.x_plus)
  BLAS.axpy!(-1.0, data.x_minus, data.x_total)
  val = f(data.x_total, data.grad_f)
  val += lambda * (sum(data.x_plus) + sum(data.x_minus))
  # g[1:dim] = grad_f + lambda * ones(dim)
  if length(g) > 0
    copy!(view(g, 1:dim), data.grad_f)
    BLAS.axpy!(lambda, ones(dim), view(g, 1:dim))
    # g[dim+1:2*dim] = -grad_f + lambda * ones(dim)
    copy!(view(g, dim+1:2*dim), data.grad_f)
    scale!(view(g, dim+1:2*dim), -1.0)
    BLAS.axpy!(lambda, ones(dim), view(g, dim+1:2*dim))
  end
  return val
end


function min_vanilla_lh(emp_data, lh_data; low_rank = false)
  dim = 2*emp_data.p*(emp_data.p-1)
  p = emp_data.p

  inner_fun(x, g) = lh(emp_data, lh_data, x, g, low_rank = low_rank)
  outer_fun(x, g) = l1_wrap(x, g, lh_data.lambda, inner_fun, lh_data)
  lb = fill(0.0, dim)
  start = mat2vec(lh_data.B0, emp_data)
  start = [start; -start]
  start[start .< 0] = 0
  (f, minx, numCall, numIter, status) = lbfgsb(outer_fun, start, lb = lb)
  B_min = vec2mat(minx[1:p*(p-1)]-minx[p*(p-1)+1:dim], emp_data)

  return B_min
end

function min_constraint_lh(emp_data,
                           data
                          )
  
  p = emp_data.p
  x_base = data.x_base
  low_rank = data.low_rank

  inner_fun(x, g) = lh(emp_data, data, x, g, low_rank = low_rank)
  outer_fun(x, g) = l1_wrap(x, g, data.lambda, inner_fun, data)
  fill!(data.lb_aug, 0.0)
  fill!(data.ub_aug, Inf)
  start = mat2vec(data.B0, emp_data)
  start = [start; -start]
  start[start .< 0] = 0

  converged = false
  dim = p * (p-1)

  function augmented_lagrangian(xaug,
                                grad,
                                rho,
                                dual,
                                data
                               )
    x = view(xaug, 1:2*dim)
    s = xaug[2*dim+1]
    if length(grad) > 0
      ret = outer_fun(x, view(grad, 1:2*dim))
    else
      ret = outer_fun(x, [])
    end
    # data.aug_diff = x - x_base
    copy!(data.aug_diff, view(x, 1:dim))
    BLAS.axpy!(-1.0, view(x, dim+1:2*dim), data.aug_diff)
    BLAS.axpy!(-1.0, x_base, data.aug_diff)
    norm_sq = vecnorm(data.aug_diff)^2
    pen = s - norm_sq + dual/rho
    ret += rho/2 * pen^2
    # data.aug_diff *= 2.0 * rho * pen
    scale!(2.0 * rho * pen, data.aug_diff)
    #= grad[1:dim] -= 2.0 .*rho.*pen.*(x - x_base) =#
    if length(grad) > 0
      BLAS.axpy!(-1.0, data.aug_diff, view(grad, 1:dim))
      BLAS.axpy!(1.0, data.aug_diff, view(grad, dim+1:2*dim))
      grad[2*dim+1] = rho * pen
    end
    return ret
  end

  function solve_auglag(x_new, x0, rho, dual, data)
    #= lb = zeros(dim+1) =#
    #= ub = zeros(dim+1) =#
    #= data.lb_aug[1:dim] = lb_x =#
    data.lb_aug[2*dim+1] = -Inf
    #= ub[1:dim] = ub_x =#
    data.ub_aug[2*dim+1] = data.upper_bound
    println("Upper bound: ", data.upper_bound)
    println("dual = ", dual)
    println("rho = ", rho)

    f_auglag(x, grad) = augmented_lagrangian(x, grad, rho, dual, data)
    (minf, minx, iters, _, status) = lbfgsb(f_auglag, x0, lb = data.lb_aug, ub = data.ub_aug, pgtol = data.final_tol, iprint=-1, factr = 1e6)
    if status == "abnormal"
      error("Lbfgsb could not terminate!")
    end
    copy!(x_new, minx)
  end

  #= x_old = zeros(dim+1) =#
  #= fill!(data.x_old, 0.0) =#
  copy!(view(data.x_old, 1:2*dim), start)
  data.x_old[2*dim+1] = data.s
  copy!(data.aug_diff, view(data.x_old, 1:dim))
  BLAS.axpy!(-1.0, view(data.x_old, dim+1:2*dim), data.aug_diff)
  BLAS.axpy!(-1.0, x_base, data.aug_diff)
  norm_old = vecnorm(data.aug_diff)^2
  constr_old = data.x_old[2*dim+1] - norm_old
  dual = data.dual
  rho = 1.0
  #= x_new = similar(x_old) =#
  println("s = ", data.x_old[2*dim+1], ", dual = ", dual)

  println()
  println("Starting augmented Lagrangian optimization")
  while ~converged
    println("Solving subproblem")
    copy!(data.x_new, data.x_old)
    solve_auglag(data.x_new, data.x_new, rho, dual, data)
    println("Subproblem solved")
    copy!(data.aug_diff, view(data.x_new, 1:dim))
    BLAS.axpy!(-1.0, view(data.x_new, dim+1:2*dim), data.aug_diff)
    BLAS.axpy!(-1.0, x_base, data.aug_diff)
    norm_new = vecnorm(data.aug_diff)^2
    constr_new = data.x_new[2*dim+1] - norm_new
    dual += rho * constr_new
    println("Dual: ", dual)
    println("Constraint: ", constr_new)

    if abs(constr_new) < data.final_tol
      converged = true
      break
    elseif abs(constr_new) > data.tau * abs(constr_old)# && rho < 10
      # Tighten penalty
      rho = data.gamma * rho 
    end

    copy!(data.x_old, data.x_new)
    constr_old = constr_new
  end

  B_min = vec2mat(data.x_new[1:dim]-data.x_new[dim+1:2*dim], emp_data)
  #= return (B_min, data.x_new[2*di+1]) =#
  data.s = copy(data.x_new[2*dim+1])
  data.dual = dual
  return B_min
end

function soft_thresh(x, r)
  return sign(x) .* max(abs(x) - r, 0)
end

function quic_old(p::Int64,
               emp_data,
               sigma::Array{Float64, 2};
               theta_prime::Array{Float64, 2} = eye(p),
               rho::Float64 = 0.0,
               lambda::Float64 = 1e-3,
               inner_mult::Float64 = 1/3,
               theta0::Array{Float64,2} = eye(Float64, p),
               tol::Float64 = 1e-4,
               inner_tol::Float64 = 1e-8,
               relaxation::Float64 = 0.2,
               search_relaxation::Float64 = 0.1,
               exponent = 4.0,
               beta::Float64 = 0.9,
               g::Array{Float64, 2} = Array{Float64,2}(0,0),
               print_stats = false
             )
  (p1, p2) = size(sigma)
  if p1 != p2
    throw(ArgumentError("sigma needs to be a square matrix"))
  end
  p = p1
  converged_outer = false
  counter = 1
  W = copy(theta0)
  Wu, Wt = similar(W), similar(W)
  LAPACK.potrf!('U', W)

  theta = copy(theta0)
  theta = (theta + theta')/2
  theta_prime = (theta_prime + theta_prime')/2
  D = similar(sigma)
  D_old = similar(sigma)
  U = similar(sigma)
  G = similar(sigma)
  Gmin = similar(sigma)
  a, b, c = 0.0, 0.0, 0.0
  theta_new = similar(sigma)
  comparison = 0.0
  l1_old = vecnorm(theta, 1)
  l1_new = 0.0
  println(size(theta))
  println(size(sigma))
  f_old = sum((theta) .* sigma) - sum(2*log(diag(W))) + rho/2 * vecnorm(theta - theta_prime)^2 + lambda * vecnorm(theta, 1)

  descent_step = false
  inner_iterations = 0

  # Calculate inverse
  LAPACK.potri!('U', W)
  W[emp_data.strict_lower_idx] = 0.0
  Wu = triu(W, 1)
  transpose!(Wt, Wu)
  W += Wt

  # Outer loop to do a Newton step
  while ~converged_outer
    if print_stats
      println("Outer iteration ", counter)
    end
    D[:] = 0.0
    U[:] = 0.0

    # Gradient of smooth part
    for I in eachindex(G)
      G[I] = sigma[I] - W[I] + rho * (theta[I] - theta_prime[I])
    end

    S = findn(triu((abs(G) .>= lambda) | (theta .!= 0)))

    # Minimum norm subgradient
    for I in eachindex(Gmin)
      if theta[I] > 0
        Gmin[I] = G[I] + lambda
      elseif theta[I] < 0
        Gmin[I] = G[I] - lambda
      else
        Gmin[I] = soft_thresh(G[I], lambda)
      end
    end

    if print_stats
      println("Gradient norm: ", vecnorm(Gmin))
    end
    if vecnorm(Gmin) < tol
      converged_outer = true
      break
    end

    # Inner loop to compute Newton direction
    r = 1
    descent_step = false
    inner_iterations = ceil(Int64, counter * inner_mult)
    inner_converged = false
    while r <= inner_iterations && ~inner_converged #|| ~descent_step
      #= if mod(r, 1) == 0 =#
        if print_stats
          println("Inner iteration ", r)
        end

        D_old[:] = D[:]
      #= end =#
      #= G[:] = theta[:] =#
      #= G[:] -= theta_prime[:] =#
      #= G[:] *= rho =#
      #= G[:] += sigma[:] =#
      #= broadcast!(-, G, G, W) =#
      #= G[:] = sigma - W + rho * (theta - theta_prime) =#
      #= G[:] = sigma - W =#
      # Determine active indices

      for (i, j) in zip(S...)
        a = W[i,j]^2
        a += rho/2
        if i != j
          a += W[i,i]*W[j,j]
        end
        b = 0.0
        @simd for k = 1:p
          @inbounds b += W[k,i] * U[j,k]
        end
        #= b += sum(W[:,i] .* U[j,:]) =#
        b += sigma[i,j]
        b -= W[i,j]
        b += rho * (theta[i,j] + D[i,j] - theta_prime[i,j])
        c = theta[i,j] + D[i,j]
        mu = -c + soft_thresh(c - b/a, lambda/a)
        if mu != 0.0
          D[i,j] += mu
          if i != j
            D[j,i] += mu
          end
          #= U[:,i] += mu * W[:,j] =#
          @simd for k = 1:p
            @inbounds U[k,i] += mu * W[k,j]
          end
          if i != j
            #= @inbounds U[:,j] += mu * W[:,i] =#
            @simd for k = 1:p
              @inbounds U[k,j] += mu * W[k,i]
            end
          end
        end
      end
      l1_new = vecnorm(theta + D, 1)
      comparison = l1_new - l1_old
      comparison *= lambda
      comparison += sum(G .* D)
      comparison *= relaxation
      #= comparison = relaxation * (sum(G .* D) + lambda * (vecnorm(theta + D, 1) - vecnorm(theta, 1))) =#
      descent_step = comparison < 0
      if print_stats
        println("descent = ", descent_step)
      end
      D_diff = vecnorm(D - D_old)
      #= println("Inner difference: ", diff) =#
      if D_diff < inner_tol
        inner_converged = true
        println("Inner converged, |D| = ", vecnorm(D), ", comparison = ", comparison)
      end
      r += 1
    end

    # Sufficient decrease condition 
    #= if comparison > -search_relaxation * vecnorm(D)^exponent =#
    #=   for I in eachindex(D) =#
    #=     D[I] = -Gmin[I] =#
    #=   end =#
    #=   #1= if print_stats =1# =#
    #=     println("Gradient step selected, ", comparison, " > ", -search_relaxation * vecnorm(D)^exponent) =#
    #=   #1= end =1# =#
    #=   comparison = lambda * (l1_new - l1_old) =#
    #=   comparison += sum(G .* D) =#
    #=   comparison *= relaxation =#
    #= end =#

    # Compute Armijo step size
    alpha = 1.0
    f_new = 0.0
    while true
      theta_new[:] = theta + alpha * D
      W[:] = copy(theta_new)
      try
        LAPACK.potrf!('U', W)
        if any(diag(W) .<= 0.0)
          throw(DomainError())
        end
      catch
        alpha *= beta
        continue
      end

      f_new = sum((theta_new) .* sigma) - sum(2*log(diag(W))) + rho/2*vecnorm(theta_new - theta_prime)^2 + lambda * vecnorm(theta_new, 1)
      #= f_new = sum((theta_new) .* sigma) - logdet(theta_new) + rho/2*vecnorm(theta_new - theta_prime)^2 + lambda * vecnorm(theta_new, 1) =#
      #= f_new = sum((theta_new) .* sigma) - logdet(theta_new) + vecnorm(theta_new, 1) =#
      if f_new <= f_old + alpha * comparison
        break
      else
        alpha *= beta
        if print_stats
          println("decrease alpha = ", alpha, ", f_new = ", f_new, ", f_old = ", f_old)
        end
        #= println("theta_new = ", theta_new) =#
        #= println("theta = ", theta) =#
      end
    end
    if print_stats
      println("alpha = ", alpha)
    end

    #= if vecnorm(theta_new - theta) < tol =#
    #=   converged_outer = true =#
    #= else =#
      counter += 1
    #= end =#

    f_old = copy(f_new)
    theta[:] = copy(theta_new)
    l1_old = copy(l1_new)
    LAPACK.potri!('U', W)
    W[emp_data.strict_lower_idx] = 0.0
    Wu[:] = W[:]
    triu!(Wu, 1)
    #= Wu = triu(W, 1) =#
    transpose!(Wt, Wu)
    W += Wt
    #= W += triu(W, 1)' =#
  end

  if length(g) > 0
    g[:] = copy(G)
  end

  return theta
end

type QuadraticPenaltyData
  lambda::Float64 # penalty parameter
  inner_mult::Float64 # how often to run the inner loop compared
                      # to the outer loop iteration
  theta0::Array{Float64,2} # Starting value
  tol::Float64 # Outer iteration tolerance
  inner_tol::Float64 # Inner iteration tolerance
  relaxation::Float64 # Multiplier in Armijo condition
  beta::Float64 # Multiply with this for Armijo
  eps::Float64 # relax free variable selection by eps
  print_stats::Bool # print additional information
  W::Array{Float64,2}
  Wu::Array{Float64,2}
  Wt::Array{Float64,2}
  theta::Array{Float64,2}
  thetaT::Array{Float64,2}
  theta_prime::Array{Float64,2}
  theta_primeT::Array{Float64,2}
  D::Array{Float64,2}
  D_old::Array{Float64,2}
  U::Array{Float64,2}
  G::Array{Float64,2}
  Gmin::Array{Float64,2}
  theta_new::Array{Float64,2}
  diff::Array{Float64,2}
  uvec::Array{Float64,1}
  hard_thresh::Float64 # Cut-off for each major iteration
end

function QuadraticPenaltyData(p, lambda, inner_mult, theta0,
                             tol, inner_tol, relaxation, beta, eps,
                             print_stats, hard_thresh,
                            )
  return QuadraticPenaltyData(
                              # For quic
                              lambda,
                              inner_mult,
                              theta0,
                              tol,
                              inner_tol,
                              relaxation,
                              beta,
                              eps,
                              print_stats,
                              zeros(p, p), # W
                              zeros(p, p), # Wu
                              zeros(p, p), # Wt
                              zeros(p, p), # theta
                              zeros(p, p), # thetaT
                              zeros(p, p), # theta_prime
                              zeros(p, p), # theta_primeT
                              zeros(p, p), # D
                              zeros(p, p), # D_old
                              zeros(p, p), # U
                              zeros(p, p), # G
                              zeros(p, p), # Gmin
                              zeros(p, p), # theta_new
                              zeros(p, p), # diff
                              zeros(p), # uvec
                              hard_thresh,
                             )
end

function QuadraticPenaltyData(p)
  return QuadraticPenaltyData(p,
                              1e-3, # lambda
                              1/3, # inner_milt
                              eye(p), # theta0
                              1e-4, # tol
                              1e-8, # inner_tol
                              0.2, # relaxation
                              0.9, # beta
                              0.01, # eps
                              false, # print_stats
                              1e-12, # hard_thresh
                             )
end

function quic(emp_data,
               quic_data,
               sigma::Array{Float64, 2};
               theta_prime::Array{Float64, 2} = eye(emp_data.p),
               rho::Float64 = 0.0,
               #= lambda::Float64 = 1e-3, =#
               #= inner_mult::Float64 = 1/3, =#
               #= theta0::Array{Float64,2} = eye(Float64, emp_data.p), =#
               #= tol::Float64 = 1e-4, =#
               #= inner_tol::Float64 = 1e-8, =#
               #= relaxation::Float64 = 0.2, =#
               #= beta::Float64 = 0.9, =#
               g::Array{Float64, 2} = Array{Float64,2}(0,0),
               #= print_stats = false =#
             )

  # Shortcuts
  data = quic_data
  lambda = data.lambda
  inner_mult = data.inner_mult
  theta0 = data.theta0
  tol = data.tol
  inner_tol = data.inner_tol
  relaxation = data.relaxation
  beta = data.beta
  eps = data.eps
  print_stats = data.print_stats

  (p1, p2) = size(sigma)
  if p1 != p2
    throw(ArgumentError("sigma needs to be a square matrix"))
  end
  p = p1
  converged_outer = false
  counter = 1
  alpha = 1.0
  copy!(data.W, theta0)
  LAPACK.potrf!('U', data.W)

  #= theta = copy(theta0) =#
  copy!(data.theta, theta0)
  symmetrize!(data.theta)
  #= transpose!(data.thetaT, data.theta) =#
  #= BLAS.axpy!(1.0, data.thetaT, data.theta) =#
  #= scale!(0.5, data.theta) =#
  #= theta = (theta0 + theta0')/2 =#
  #= transpose!(data.theta_primeT, theta_prime) =#
  copy!(data.theta_prime, theta_prime)
  #= BLAS.axpy!(1.0, data.theta_primeT, data.theta_prime) =#
  #= scale!(0.5, data.theta_prime) =#
  #= theta_prime = (theta_prime + theta_prime')/2 =#
  symmetrize!(data.theta_prime)

  #= D = similar(sigma) =#
  #= D_old = similar(sigma) =#
  #= U = similar(sigma) =#
  #= G = similar(sigma) =#
  #= Gmin = similar(sigma) =#
  a, b, c = 0.0, 0.0, 0.0
  #= theta_new = similar(sigma) =#
  comparison = 0.0
  l1_old = vecnorm(data.theta, 1)
  l1_new = 0.0
  copy!(data.diff, data.theta)
  BLAS.axpy!(-1.0, data.theta_prime, data.diff)
  #= f_old = sum((data.theta) .* sigma) - sum(2*log(diag(data.W))) + rho/2 * vecnorm(data.theta - data.theta_prime)^2 + lambda * vecnorm(data.theta, 1) =#
  f_old = BLAS.dot(p^2, data.theta, 1, sigma, 1) - sum(2*log(diag(data.W))) + rho/2 * vecnorm(data.diff)^2 + lambda * vecnorm(data.theta, 1)

  descent_step = false
  inner_iterations = 0

  # Calculate inverse
  LAPACK.potri!('U', data.W)
  data.W[emp_data.strict_lower_idx] = 0.0
  copy!(data.Wu, data.W)
  triu!(data.Wu, 1)
  #= Wu = triu(W, 1) =#
  transpose!(data.Wt, data.Wu)
  #= W += Wt =#
  BLAS.axpy!(1.0, data.Wt, data.W)

  # Outer loop to do a Newton step
  while ~converged_outer
    if print_stats
      println("Outer iteration ", counter)
    end
    #= D[:] = 0.0 =#
    fill!(data.D, 0.0)
    #= U[:] = 0.0 =#
    fill!(data.U, 0.0)

    # Gradient of smooth part
    # data.diff = data.theta - data.theta_prime
    #= for I in eachindex(data.G) =#
    #=   data.G[I] = sigma[I] - data.W[I] + rho * (data.theta[I] - data.theta_prime[I]) =#
    #= end =#
    copy!(data.diff, data.theta)
    BLAS.axpy!(-1.0, data.theta_prime, data.diff)
    copy!(data.G, sigma)
    BLAS.axpy!(-1.0, data.W, data.G)
    BLAS.axpy!(rho, data.diff, data.G)

    S = findn(triu((abs(data.G) .>= lambda - eps) | (data.theta .!= 0)))

    # Minimum norm subgradient
    @simd for I in eachindex(data.Gmin)
      if data.theta[I] > 0
        data.Gmin[I] = data.G[I] + lambda
      elseif data.theta[I] < 0
        data.Gmin[I] = data.G[I] - lambda
      else
        data.Gmin[I] = soft_thresh(data.G[I], lambda)
      end
    end

    if print_stats
      println("Gradient norm: ", vecnorm(data.Gmin))
    end
    if vecnorm(data.Gmin, 1)*alpha < tol*vecnorm(data.theta, 1)
      converged_outer = true
      break
    end

    # Inner loop to compute Newton direction
    r = 1
    descent_step = false
    inner_iterations = floor(Int64, 1 + counter * inner_mult)
    inner_converged = false
    while r <= inner_iterations && ~inner_converged #|| ~descent_step
    #= while r <= 100 =#
      #= if mod(r, 1) == 0 =#
        if print_stats
          println("Inner iteration ", r)
        end

        copy!(data.D_old, data.D)
        #= D_old[:] = D[:] =#
      #= end =#
      #= G[:] = theta[:] =#
      #= G[:] -= theta_prime[:] =#
      #= G[:] *= rho =#
      #= G[:] += sigma[:] =#
      #= broadcast!(-, G, G, W) =#
      #= G[:] = sigma - W + rho * (theta - theta_prime) =#
      #= G[:] = sigma - W =#
      # Determine active indices

      for (i, j) in zip(S...)
        a = data.W[i,j]^2
        a += rho
        if i != j
          a += data.W[i,i]*data.W[j,j]
        end
        b = 0.0
        @simd for k = 1:p
          @inbounds b += data.W[k,i] * data.U[j,k]
        end
        #= b += sum(W[:,i] .* U[j,:]) =#
        #= copy!(data.uvec, view(data.U, j, :)) =#
        #= b += BLAS.dot(view(data.W, :, i), data.uvec) =#
        b += sigma[i,j]
        b -= data.W[i,j]
        b += rho * (data.theta[i,j] + data.D[i,j] - data.theta_prime[i,j])
        c = data.theta[i,j] + data.D[i,j]

        #= # DEBUG start =#
        #= mu = 1 =#
        #= f0 = sum((sigma - data.W).*data.D) + 0.5*trace(data.W*data.D*data.W*data.D) + rho/2*vecnorm(data.theta + data.D - data.theta_prime)^2 =#
        #= data.D[i,j] += mu =#
        #= if i != j =#
        #=   data.D[j,i] += mu =#
        #= end =#
        #= f_plus = sum((sigma - data.W).*data.D) + 0.5*trace(data.W*data.D*data.W*data.D) + rho/2*vecnorm(data.theta + data.D - data.theta_prime)^2 =#
        #= data.D[i,j] -= 2*mu =#
        #= if i != j =#
        #=   data.D[j,i] -= 2*mu =#
        #= end =#
        #= f_min = sum((sigma - data.W).*data.D) + 0.5*trace(data.W*data.D*data.W*data.D) + rho/2*vecnorm(data.theta + data.D - data.theta_prime)^2 =#
        #= #1= c_ideal = f0 =1# =#
        #= b_ideal = (f_plus - f_min)/(2*mu) =#
        #= a_ideal = 2*(f_plus - mu*b_ideal - f0)/(mu^2) =#
        #= if i != j =#
        #=   b_ideal /= 2 =#
        #=   a_ideal /= 2 =#
        #= end =#
        #= #1= if i == j =1# =#
        #= if counter == 1 =#
        #=   println("-------------------------------------------") =#
        #=   println("Theory: a = ", a_ideal, ", b = ", b_ideal) =#
        #=   println("Practice: a = ", a, ", b = ", b) =#
        #= end =#
        #= #1= end =1# =#
        #= data.D[i,j] += mu =#
        #= if i != j =#
        #=   data.D[j,i] += mu =#
        #= end =#
        #= # DEBUG end =#

        mu = -c + soft_thresh(c - b/a, lambda/a)
        if mu != 0.0
          data.D[i,j] += mu
          if i != j
            data.D[j,i] += mu
          end
          #= U[:,i] += mu * W[:,j] =#
          @simd for k = 1:p
            @inbounds data.U[k,i] += mu * data.W[k,j]
          end
          #= BLAS.axpy!(mu, view(data.W, :, j), view(data.U, :, i)) =#
          #= BLAS.axpy!(mu, data.W[:, j], data.U[:,i]) =#
          if i != j
            #= @inbounds U[:,j] += mu * W[:,i] =#
            @simd for k = 1:p
              @inbounds data.U[k,j] += mu * data.W[k,i]
            end
            #= BLAS.axpy!(mu, view(data.W, :, i), view(data.U, :, j)) =#
          end
        end
      end
      #= l1_new = vecnorm(data.theta + data.D, 1) =#
      copy!(data.diff, data.theta)
      BLAS.axpy!(1.0, data.D, data.diff)
      l1_new = vecnorm(data.diff, 1)
      comparison = l1_new - l1_old
      comparison *= lambda
      #= comparison += sum(data.G .* data.D) =#
      comparison += BLAS.dot(p^2, data.G, 1, data.D, 1)
      comparison *= relaxation
      #= comparison = relaxation * (sum(G .* D) + lambda * (vecnorm(theta + D, 1) - vecnorm(theta, 1))) =#
      descent_step = comparison < 0
      if print_stats
        println("descent = ", descent_step)
      end
      #= diff = vecnorm(data.D - data.D_old) =#
      copy!(data.diff, data.D)
      BLAS.axpy!(-1.0, data.D_old, data.diff)
      diff = vecnorm(data.diff)
      #= println("Inner difference: ", diff) =#
      if diff < inner_tol
        inner_converged = true
        #= println("Inner converged, |D| = ", vecnorm(data.D), ", comparison = ", comparison) =#
      end
      r += 1
    end

    #= global first_pass =#
    #= if first_pass && counter >= 2000 =#
    #=   println("Sticky!") =#
    #=   println("theta_prime = ", theta_prime) =#
    #=   save("sticky_data2.jld", "D", data.D, "theta", data.theta, "sigma", sigma, "rho", rho, "lambda", lambda, "theta_prime", data.theta_prime, "Gmin", data.Gmin) =#
    #=   println(S) =#
    #=   first_pass = false =#
    #= end =#


    # Sufficient decrease condition 
    #= if comparison > -search_relaxation * vecnorm(D)^exponent =#
    #=   for I in eachindex(D) =#
    #=     D[I] = -Gmin[I] =#
    #=   end =#
    #=   #1= if print_stats =1# =#
    #=     println("Gradient step selected, ", comparison, " > ", -search_relaxation * vecnorm(D)^exponent) =#
    #=   #1= end =1# =#
    #=   comparison = lambda * (l1_new - l1_old) =#
    #=   comparison += sum(G .* D) =#
    #=   comparison *= relaxation =#
    #= end =#

    # Compute Armijo step size
    alpha = 1.0
    f_new = 0.0
    while true
      #= data.theta_new[:] = data.theta + alpha * data.D =#
      copy!(data.theta_new, data.theta)
      BLAS.axpy!(alpha, data.D, data.theta_new)
      #= W[:] = copy(theta_new) =#
      copy!(data.W, data.theta_new)
      try
        LAPACK.potrf!('U', data.W)
        if any(diag(data.W) .<= 0.0)
          throw(DomainError())
        end
      catch
        alpha *= beta
        continue
      end

      copy!(data.diff, data.theta_new)
      BLAS.axpy!(-1.0, data.theta_prime, data.diff)
      f_new = BLAS.dot(p^2, data.theta_new, 1, sigma, 1) - sum(2*log(diag(data.W))) + rho/2 * vecnorm(data.diff)^2 + lambda * vecnorm(data.theta_new, 1)
      #= f_new = sum((data.theta_new) .* sigma) - sum(2*log(diag(data.W))) + rho/2*vecnorm(data.theta_new - data.theta_prime)^2 + lambda * vecnorm(data.theta_new, 1) =#
      #= f_new = sum((theta_new) .* sigma) - logdet(theta_new) + rho/2*vecnorm(theta_new - theta_prime)^2 + lambda * vecnorm(theta_new, 1) =#
      #= f_new = sum((theta_new) .* sigma) - logdet(theta_new) + vecnorm(theta_new, 1) =#
      if f_new <= f_old + alpha * comparison
        break
      else
        alpha *= beta
        if print_stats
          #= print("theta = ") =#
          #= println(data.theta) =#
          #= print("D = ") =#
          #= println(data.D) =#
          #= print("sigma = ") =#
          #= println(sigma) =#
          #= print("theta_prime = ") =#
          #= println(theta_prime) =#
          #= print("rho = ") =#
          #= println(rho) =#
          #= print("lambda = ") =#
          #= println(lambda) =#
          println("decrease alpha = ", alpha, ", f_new = ", f_new, ", f_old = ", f_old)
        end
        #= println("theta_new = ", theta_new) =#
        #= println("theta = ", theta) =#
      end
    end
    if print_stats
      println("alpha = ", alpha)
    end

    #= if vecnorm(theta_new - theta) < tol =#
    #=   converged_outer = true =#
    #= else =#
      counter += 1
    #= end =#

    f_old = copy(f_new)
    #= theta[:] = copy(theta_new) =#
    copy!(data.theta, data.theta_new)
    for i = 1:p
      for j = setdiff(1:p, i)
        data.theta[i,j] = hard_thresh(data.theta[i,j], data.hard_thresh)
      end
    end
    l1_old = copy(l1_new)
    LAPACK.potri!('U', data.W)
    data.W[emp_data.strict_lower_idx] = 0.0
    #= data.Wu[:] = data.W[:] =#
    copy!(data.Wu, data.W)
    triu!(data.Wu, 1)
    #= Wu = triu(W, 1) =#
    transpose!(data.Wt, data.Wu)
    #= W += Wt =#
    BLAS.axpy!(1.0, data.Wt, data.W)
    #= W += triu(W, 1)' =#
  end

  if length(g) > 0
    #= g[:] = copy(G) =#
    copy!(g, data.G)
  end

  return data.theta
end

# First version
function admm_old(emp_data, solvers, dual_updates, vars0; compute_bmap = [], rho = 1.0, tol = 1e-6, bb = false, dual_balancing = false, mu = 10.0, tau = 2.0)

  p = emp_data.p

  converged = false
  vars = deepcopy(vars0)
  vars_old = deepcopy(vars0)
  duals = []
  bmap_old = compute_bmap(vars, rho)
  bmap = copy(bmap_old)
  counter = 1
  for e = 1:emp_data.E
    push!(duals, zeros(p, p))
  end
  primal_residuals = deepcopy(duals)

  while ~converged
    # Minimization steps
    for s! in solvers
      s!(vars, duals, rho)
    end

    # Dual updates
    res_norm = 0.0
    for (dual, dual_update, pres) in zip(duals, dual_updates, primal_residuals)
      pres[:] = dual_update(vars)
      res_norm += vecnorm(pres)^2
      dual[:] = dual + rho * pres
      dual[:] = (dual + dual')/2
    end
    res_norm = sqrt(res_norm)
    bmap = compute_bmap(vars, rho)
    dual_res = rho * (bmap - bmap_old) # this is not true in general
    dual_res_norm = vecnorm(dual_res)
    bmap_old = copy(bmap)

    # Check for convergence
    println("Norm difference: ", vecnorm(vars - vars_old))
    println("Primal residual: ", res_norm)
    println("Dual residual: ", dual_res_norm)
    #= if vecnorm(vars - vars_old) < tol =#
    if res_norm < tol && dual_res_norm < tol
      converged = true
    else
      println("Iteration ", counter)
      counter += 1
    end

    # Residual balancing
    if dual_balancing
      if res_norm > mu * dual_res_norm
        rho *= tau
      elseif dual_res_norm > mu * res_norm
        rho /= tau
      end
    end

    vars_old = deepcopy(vars)
    println("rho = ", rho)
  end

  return vars
end

type ConstraintData_old
  B::Array{Float64,2}
  Breds::Array{Array{Float64,2},1}
  Bredst::Array{Array{Float64,2},1}
  Bmuldiff::Array{Array{Float64,2},1}
  Ainv::Array{Float64,2}
  Ainv2::Array{Float64,2}
  diff::Array{Float64,2}
  diff_sum::Array{Float64,2}
  Gmat::Array{Float64,2}
  Gmat_sum::Array{Float64,2}
  initialize_array::Bool
end

function ConstraintData_old(p)
  return ConstraintData_old(zeros(p, p), [], [], [], zeros(p,p), zeros(p,p), zeros(p,p), zeros(p,p), zeros(p,p), zeros(p,p), true)
end

function constr_least_sq_old(x, g, vars, duals, rho, emp_data, data::ConstraintData_old, Us_ind = Us_ind, Js_ind = Js_ind; low_rank = false)
  """
  Calculate the least squares function linking B and the thetas, for use with lbfgsb
  """

  # Shorthands
  E = emp_data.E
  find_offdiag_idx = emp_data.find_offdiag_idx

  compute_grad = length(g) > 0
  data.initialize_array = length(data.Breds) == 0
  val = 0.0
  fill!(data.B, 0.0)
  vec2mat(x, emp_data, inplace = data.B)
 
  if compute_grad
    fill!(g, 0.0)
    if low_rank
      fill!(data.diff_sum, 0.0)
      fill!(data.Gmat_sum, 0.0)
    end
  end
 
  (p1, p2) = size(data.Ainv)
  if low_rank
    # Assemble (I - B)
    fill!(data.Ainv, 0.0)
    for i = 1:p1
      data.Ainv[i,i] = 1.0
    end

    data.Ainv[:] .-= data.B[:]
    BLAS.gemm!('T', 'N', 1.0, data.Ainv, data.Ainv, 0.0, data.Ainv2)
  end

  for e = 1:E
    fill!(data.diff, 0.0)
    if low_rank
      if data.initialize_array
        push!(data.Breds, zeros(length(Js_ind[e]), p2))
        push!(data.Bmuldiff, zeros(length(Js_ind[e]), p2))
        push!(data.Bredst, zeros(p1, length(Js_ind[e])))
      end
      copy!(data.Breds[e], view(data.B, Js_ind[e], :))
      transpose!(data.Bredst[e], data.Breds[e])
      copy!(data.diff, data.Ainv2)
      scale!(data.diff, -1.0)
      BLAS.axpy!(-1.0, data.Breds[e], view(data.diff, Js_ind[e], :))
      BLAS.axpy!(-1.0, data.Bredst[e], view(data.diff, :, Js_ind[e]))
      #= println(size(data.Bredst[e]), ", ", size(data.Breds[e])) =#
      BLAS.gemm!('N', 'N', 1.0, data.Bredst[e], data.Breds[e], 1.0, data.diff)
    else
      fill!(data.Ainv, 0.0)
      for i = 1:p1
        data.Ainv[i, i] = 1
      end

      #= @inbounds for col in 1:p2 =#
      #=   @simd for row in Us_ind[e] =#
      #=     data.Ainv[row, col] -= data.B[row, col] =#
      #=   end =#
      #= end =#
      BLAS.axpy!(-1.0, view(data.B, Us_ind[e], :), view(data.Ainv, Us_ind[e], :))

      #= Ainv = eye(p) - Us[e]*data.B =#
      #= println("Difference in Ainv: ", vecnorm(Ainv - data.Ainv)) =#
      #= println(Ainv) =#
      #= println(data.Ainv) =#
      #= data.Ainv = Ainv =#

      #= diff = vars[e] - Ainv'*Ainv + duals[e]/rho =#
      BLAS.gemm!('T', 'N', -1.0, data.Ainv, data.Ainv, 0.0, data.diff)
      #= data.diff[:] = -data.Ainv'*data.Ainv =#
      #= @inbounds @simd for I in eachindex(data.diff) =#
      #=   data.diff[I] += vars[e][I] + duals[e][I]/rho =#
      #= end =#
    end
    BLAS.axpy!(1.0, vars[e], data.diff)
    BLAS.axpy!(1.0/rho, duals[e], data.diff)
    #= data.diff .+= vars[e] + duals[e]/rho =#
    #= val += vecnorm(diff)^2 =#
    val += vecnorm(data.diff)^2
    if compute_grad
      fill!(data.Gmat, 0.0)
      if low_rank
        BLAS.axpy!(1.0, data.diff, data.diff_sum)
        BLAS.axpy!(-2.0, view(data.diff, Js_ind[e], :), view(data.Gmat_sum, Js_ind[e], :))
        BLAS.gemm!('N', 'N', 2.0, data.Breds[e], data.diff, 0.0, data.Bmuldiff[e])
        BLAS.axpy!(1.0, data.Bmuldiff[e], view(data.Gmat_sum, Js_ind[e], :))
      else
        BLAS.gemm!('N', 'N', 2.0, data.Ainv, data.diff, 0.0, data.Gmat)
        data.Gmat[Js_ind[e],:] = 0.0
        BLAS.axpy!(1.0, view(data.Gmat, find_offdiag_idx), g)
      end
    end
  end
  data.initialize_array = false

  if compute_grad && low_rank
    BLAS.gemm!('N', 'N', 2.0, data.Ainv, data.diff_sum, 1.0, data.Gmat_sum)
    copy!(g, view(data.Gmat_sum, find_offdiag_idx))
  end

  val *= rho/2
  if compute_grad
    scale!(g, rho)
  end
 
  #= println("Val = ", val) =#
  #= println("g = ", g) =#
  return val
end

type ADMMData_old
  quic_data
  constr_data
  B::Array{Float64,2}
  theta_prime::Array{Float64,2}
end

function ADMMData_old(emp_data, quic_data, constr_data)
  return ADMMData_old(
                  QuadraticPenaltyData(emp_data.p),
                  constr_data,
                  zeros(emp_data.p, emp_data.p),
                  zeros(emp_data.p, emp_data.p)
                 )
end

function min_admm_old(emp_data, admm_data, lambda, B0, rho)

  # Shorthands for variables
  E = emp_data.E
  p = emp_data.p
  sigmas_emp = emp_data.sigmas_emp
  B = admm_data.B
  theta_prime = admm_data.theta_prime
  I = emp_data.I
  Us = emp_data.Us
  theta_prime = admm_data.theta_prime
  qu_data = admm_data.quic_data

  function solve_ml_quic!(vars, duals, rho, sigmas, e)
    #= println("Running quic for ", e) =#
    vec2mat(vars[end], emp_data, inplace = B)
    theta_prime[:] = (I - Us[e]*B)'*(I - Us[e]*B) - duals[e]/rho
    theta = quic(emp_data, qu_data, sigmas[e],
                  theta_prime = theta_prime, rho = rho, lambda = lambda,
                  theta0 = vars[e], tol = 1e-2, inner_tol = 1e-6)
    copy!(vars[e], theta)
  end

  #= constr_data = ConstraintData(p) =#
  constr_data = admm_data.constr_data

  function solve_constr!(vars, duals, rho)
    function lbfgs_obj(x, g)
      ret = constr_least_sq_old(x, g, vars, duals, rho, emp_data, constr_data, emp_data.Us_ind, emp_data.Js_ind, low_rank = false)
      return ret
    end
    #= Profile.clear() =#
    #= @profile (minf, minx, ret) = Liblbfgs.lbfgs(vars[end], lbfgs_obj, print_stats = false) =#
    (minf, minx, ret) = lbfgsb(lbfgs_obj, vars[end])
    vars[end][:] = minx
  end

  function compute_bmap(vars, rho)
    B = vec2mat(vars[end], emp_data)
    ret = zeros(p^2*E)
    for e = 1:E
      Ainv = I - emp_data.Us[e]*B
      ret[1 + (e-1)*p^2:e*p^2] = Ainv'*Ainv
    end
    return ret
  end

  function dual_update(emp_data, vars, e)
    B = vec2mat(vars[end], emp_data)
    Ainv = I - emp_data.Us[e] * B
    return vars[e] - Ainv' * Ainv
  end

  solvers = []
  compute_grads_h = []
  compute_grads_g = []
  for e = 1:E
    push!(solvers, (vars, duals, rho) -> solve_ml_quic!(vars, duals, rho, sigmas_emp, e))
    #= push!(compute_grads_h, (g, vars) -> compute_grad_h!(g, vars, e)) =#
    #= push!(compute_grads_g, (g, vars) -> compute_grad_g!(g, vars, e)) =#
  end

  vars = []
  duals = []
  for e = 1:E
    push!(vars, inv(sigmas_emp[e]))
    #= dual = 0.1*randn(p, p) =#
    dual = zeros(p, p)
    dual = (dual + dual')/2
    push!(duals, dual)
  end
  #= B0 = 100 * triu(randn(p,p), 1) =#
  push!(vars, mat2vec(B0, emp_data))
  #= push!(vars, zeros(p*(p-1))) =#

  #= solve_constr!(vars, duals, rho) =#
  #= B_constr = vec2mat(vars[end]) =#
  #= println(vecnorm(vars[end] - mat2vec(B))) =#

  dual_updates = []
  for e = 1:E
    push!(dual_updates, vars -> dual_update(emp_data, vars, e))
  end
  push!(solvers, solve_constr!)

  vars_result = admm_old(emp_data, solvers, dual_updates, vars, rho = rho, tol = 1e-2, compute_bmap = compute_bmap, dual_balancing = true, mu = 5.0, tau = 1.5)
  B_admm = vec2mat(vars_result[end], emp_data)
  #= println(vecnorm(vars_result[end] - mat2vec(B))) =#
  return B_admm
end

# Second version
function admm(emp_data, admm_data, solvers, dual_update, vars0; compute_bmap = [])

  # Shortcuts
  p = emp_data.p
  rho = admm_data.rho
  tol_abs = admm_data.tol_abs
  tol_rel = admm_data.tol_rel
  dual_balancing = admm_data.dual_balancing
  bb = admm_data.bb
  tighten = admm_data.tighten
  mu = admm_data.mu
  tau = admm_data.tau
  duals = admm_data.duals
  T = admm_data.T
  eps_cor = admm_data.eps_cor

  converged = false
  vars = deepcopy(vars0)
  vars_old = deepcopy(vars0)
  # Init duals
  #= duals = [] =#
  #= for e = 1:emp_data.E =#
  #=   push!(duals, zeros(p, p)) =#
  #= end =#
  primal_residuals = deepcopy(duals)

  if bb
    duals_old = deepcopy(duals)
    duals_hat = deepcopy(duals)
    duals_hat_old = deepcopy(duals_hat)
    Ainv2s_old = deepcopy(admm_data.Ainv2s)
    thetas_old = []
    for e = 1:length(duals)
      push!(thetas_old, copy(vars[e]))
    end
    delta_thetas = deepcopy(thetas_old)
    delta_Ainv2s = deepcopy(admm_data.Ainv2s)
    delta_duals = deepcopy(duals)
    delta_duals_hat = deepcopy(duals)
    first_run = true
  end

  if bb
    primal_residuals_hat = deepcopy(duals)
  end

  dual_update(admm_data, vars, primal_residuals, compute_B_only = true)
  bmap_old = fill(0.0, emp_data.E*p^2)
  compute_bmap(admm_data, vars, rho, bmap_old)
  bmap = copy(bmap_old)
  counter = 1

  dim_primal = sum([length(var) for var in vars])
  dim_dual = sum([length(dual) for dual in duals])
  println("tol_primal_abs = ", sqrt(dim_primal) * tol_abs, ", tol_dual_abs = ", sqrt(dim_dual) * tol_abs)

  push!(admm_path, [])
  while ~converged
    status = Dict()
    # Minimization steps
    for s! in solvers
      s!(vars, duals, rho)
    end

    # Dual updates
    res_norm = 0.0

    if bb && mod(counter, T) == 1
      for e = 1:length(duals)
        duals_hat[e] = duals[e] + rho * (vars[e] - admm_data.Ainv2s[e])
      end
    end

    dual_update(admm_data, vars, primal_residuals)
    for e = 1:length(duals)
      pres = primal_residuals[e]
      res_norm += vecnorm(pres)^2
      dual = duals[e]
      dual[:] = dual + rho * pres
      #= dual[:] = (dual + dual')/2 =#
      symmetrize!(dual)
    end

    if bb && mod(counter, T) == 1 && !first_run
      delta_duals_hat_sq = 0.0
      delta_thetas_mul_delta_duals_hat = 0.0
      delta_thetas_sq = 0.0

      delta_duals_sq = 0.0
      delta_Ainv2s_mul_delta_duals = 0.0
      delta_Ainv2s_sq = 0.0

      for e = 1:length(duals)
        delta_thetas[e] = vars[e] - thetas_old[e]
        delta_Ainv2s[e] = admm_data.Ainv2s[e] - Ainv2s_old[e]
        delta_duals[e] = -(duals[e] - duals_old[e])
        delta_duals_hat[e] = -(duals_hat[e] - duals_hat_old[e])

        delta_duals_hat_sq += sum(delta_duals_hat[e].^2)
        delta_thetas_mul_delta_duals_hat += sum(delta_thetas[e] .* delta_duals_hat[e])
        delta_thetas_sq += sum(delta_thetas[e].^2)

        delta_duals_sq += sum(delta_duals[e].^2)
        delta_Ainv2s_mul_delta_duals += sum(delta_duals[e] .* delta_Ainv2s[e])
        delta_Ainv2s_sq += sum(delta_Ainv2s[e].^2)
      end

      alpha_sd = delta_duals_hat_sq / delta_thetas_mul_delta_duals_hat
      alpha_mg = delta_thetas_mul_delta_duals_hat / delta_thetas_sq

      beta_sd = delta_duals_sq / delta_Ainv2s_mul_delta_duals
      beta_mg = delta_Ainv2s_mul_delta_duals / delta_Ainv2s_sq

      #= alpha = 2 * alpha_mg > alpha_sd ? alpha_mg : alpha_sd - alpha_mg / 2 =#
      #= beta = 2 * beta_mg > beta_sd ? beta_mg : beta_sd - beta_mg / 2 =#
      alpha = alpha_sd
      beta = beta_sd

      alpha_cor = delta_thetas_mul_delta_duals_hat / (sqrt(delta_thetas_sq) * sqrt(delta_duals_hat_sq))
      beta_cor = delta_Ainv2s_mul_delta_duals / (sqrt(delta_Ainv2s_sq) * sqrt(delta_duals_sq))

      println("alpha_sd = ", alpha_sd, ", alpha_mg = ", alpha_mg, ", beta_sd = ", beta_sd, ", beta_mg = ", beta_mg)
      println("alpha = ", alpha, ", beta = ", beta, ", alpha_cor = ", alpha_cor, ", beta_cor = ", beta_cor)

      if alpha_cor > eps_cor && beta_cor > eps_cor
        rho = sqrt(alpha * beta)
      elseif alpha_cor > eps_cor && beta_cor <= eps_cor
        rho = alpha
      elseif alpha_cor <= eps_cor && beta_cor > eps_cor
        rho = beta
      end
    end

    if bb && mod(counter, T) == 1
      for e = 1:length(duals)
        duals_hat_old[e] = copy(duals_hat[e])
        thetas_old[e] = copy(vars[e])
        duals_old[e] = copy(duals[e])
        Ainv2s_old[e] = copy(admm_data.Ainv2s[e])
      end
      B_old = copy(vars[end])
      first_run = false
    end

    #= for (dual, dual_update, pres) in zip(duals, dual_updates, primal_residuals) =#
    #=   pres[:] = dual_update(vars) =#
    #=   res_norm += vecnorm(pres)^2 =#
    #=   dual[:] = dual + rho * pres =#
    #=   dual[:] = (dual + dual')/2 =#
    #= end =#
    res_norm = sqrt(res_norm)
    compute_bmap(admm_data, vars, rho, bmap)
    #= bmap = compute_bmap(vars, rho) =#
    dual_res = rho * (bmap - bmap_old) # this is not true in general
    dual_res_norm = vecnorm(dual_res)
    bmap_old = copy(bmap)

    # Check for convergence
    println("Norm difference: ", vecnorm(vars - vars_old))
    println("Primal residual: ", res_norm)
    println("Dual residual: ", dual_res_norm)
    #= if vecnorm(vars - vars_old) < tol =#
    tol_primal = sqrt(dim_primal) * tol_abs + max(sqrt(sum([vecnorm(var) for var in vars])), vecnorm(bmap)) * tol_rel
    tol_dual = sqrt(dim_dual) * tol_abs + sqrt(sum([vecnorm(dual) for dual in duals])) * tol_rel
    #= println("tol_primal = ", tol_primal, ", tol_dual = ", tol_dual) =#
    if res_norm < tol_primal && dual_res_norm < tol_dual
      converged = true
    else
      println("Iteration ", counter)
      counter += 1
    end

    # Residual balancing
    if dual_balancing
      if res_norm > mu * dual_res_norm
        rho *= tau
      elseif dual_res_norm > mu * res_norm
        rho /= tau
      end
    end

    # Tightening
    if tighten && counter >= 3
      if res_norm > 1.5 * res_norm_old
        rho *= 2
      end
    end

    res_norm_old = copy(res_norm)

    global admm_path
    status["vars"] = deepcopy(vars)
    status["duals"] = deepcopy(duals)
    status["bmap"] = deepcopy(bmap)
    status["rho"] = copy(rho)
    push!(admm_path[end], status)

    #= if (counter <= 40 || admm_data.quic_data.lambda > 0.005) && mod(counter, 20) == 0 =#
    #=   println("Serializing") =#
    #=   savevar(debugfname, admm_path) =#
    #= end =#

    println("rho = ", rho)
    vars_old = deepcopy(vars)
  end

  return vars
end

type ConstraintData
  B::Array{Float64,2}
  Breds::Array{Array{Float64,2},1}
  Bredst::Array{Array{Float64,2},1}
  Bmuldiff::Array{Array{Float64,2},1}
  Ainv::Array{Float64,2}
  Ainv2::Array{Float64,2}
  diff::Array{Float64,2}
  diff_sum::Array{Float64,2}
  Gmat::Array{Float64,2}
  Gmat_sum::Array{Float64,2}
  initialize_array::Bool
end

function ConstraintData(p)
  return ConstraintData(zeros(p, p), [], [], [], zeros(p,p), zeros(p,p), zeros(p,p), zeros(p,p), zeros(p,p), zeros(p,p), true)
end

function constr_least_sq(x, g, vars, duals, rho, emp_data, data::ConstraintData, Us_ind = Us_ind, Js_ind = Js_ind; low_rank = false)
  """
  Calculate the least squares function linking B and the thetas, for use with lbfgsb
  """

  # Shorthands
  E = emp_data.E
  find_offdiag_idx = emp_data.find_offdiag_idx

  compute_grad = length(g) > 0
  data.initialize_array = length(data.Breds) == 0
  val = 0.0
  fill!(data.B, 0.0)
  vec2mat(x, emp_data, inplace = data.B)
 
  if compute_grad
    fill!(g, 0.0)
    if low_rank
      fill!(data.diff_sum, 0.0)
      fill!(data.Gmat_sum, 0.0)
    end
  end
 
  (p1, p2) = size(data.Ainv)
  if low_rank
    # Assemble (I - B)
    fill!(data.Ainv, 0.0)
    for i = 1:p1
      data.Ainv[i,i] = 1.0
    end

    data.Ainv[:] .-= data.B[:]
    BLAS.gemm!('T', 'N', 1.0, data.Ainv, data.Ainv, 0.0, data.Ainv2)
    symmetrize!(data.Ainv2)
  end

  for e = 1:E
    fill!(data.diff, 0.0)
    if low_rank
      if data.initialize_array
        push!(data.Breds, zeros(length(Js_ind[e]), p2))
        push!(data.Bmuldiff, zeros(length(Js_ind[e]), p2))
        push!(data.Bredst, zeros(p1, length(Js_ind[e])))
      end
      copy!(data.Breds[e], view(data.B, Js_ind[e], :))
      transpose!(data.Bredst[e], data.Breds[e])
      copy!(data.diff, data.Ainv2)
      scale!(data.diff, -1.0)
      BLAS.axpy!(-1.0, data.Breds[e], view(data.diff, Js_ind[e], :))
      BLAS.axpy!(-1.0, data.Bredst[e], view(data.diff, :, Js_ind[e]))
      #= println(size(data.Bredst[e]), ", ", size(data.Breds[e])) =#
      BLAS.gemm!('N', 'N', 1.0, data.Bredst[e], data.Breds[e], 1.0, data.diff)
      symmetrize!(data.diff)
    else
      fill!(data.Ainv, 0.0)
      for i = 1:p1
        data.Ainv[i, i] = 1
      end

      #= @inbounds for col in 1:p2 =#
      #=   @simd for row in Us_ind[e] =#
      #=     data.Ainv[row, col] -= data.B[row, col] =#
      #=   end =#
      #= end =#
      BLAS.axpy!(-1.0, view(data.B, Us_ind[e], :), view(data.Ainv, Us_ind[e], :))

      #= Ainv = eye(p) - Us[e]*data.B =#
      #= println("Difference in Ainv: ", vecnorm(Ainv - data.Ainv)) =#
      
      #= println(Ainv) =#
      #= println(data.Ainv) =#
      #= data.Ainv = Ainv =#

      #= diff = vars[e] - Ainv'*Ainv + duals[e]/rho =#
      BLAS.gemm!('T', 'N', -1.0, data.Ainv, data.Ainv, 0.0, data.diff)
      symmetrize!(data.diff)
      #= data.diff[:] = -data.Ainv'*data.Ainv =#
      #= @inbounds @simd for I in eachindex(data.diff) =#
      #=   data.diff[I] += vars[e][I] + duals[e][I]/rho =#
      #= end =#
    end
    BLAS.axpy!(1.0, vars[e], data.diff)
    BLAS.axpy!(1.0/rho, duals[e], data.diff)
    #= data.diff .+= vars[e] + duals[e]/rho =#
    #= val += vecnorm(diff)^2 =#
    val += vecnorm(data.diff)^2
    if compute_grad
      fill!(data.Gmat, 0.0)
      if low_rank
        BLAS.axpy!(1.0, data.diff, data.diff_sum)
        BLAS.axpy!(-2.0, view(data.diff, Js_ind[e], :), view(data.Gmat_sum, Js_ind[e], :))
        BLAS.gemm!('N', 'N', 2.0, data.Breds[e], data.diff, 0.0, data.Bmuldiff[e])
        BLAS.axpy!(1.0, data.Bmuldiff[e], view(data.Gmat_sum, Js_ind[e], :))
      else
        BLAS.gemm!('N', 'N', 2.0, data.Ainv, data.diff, 0.0, data.Gmat)
        data.Gmat[Js_ind[e],:] = 0.0
        BLAS.axpy!(1.0, view(data.Gmat, find_offdiag_idx), g)
      end
    end
  end
  data.initialize_array = false

  if compute_grad && low_rank
    BLAS.gemm!('N', 'N', 2.0, data.Ainv, data.diff_sum, 1.0, data.Gmat_sum)
    copy!(g, view(data.Gmat_sum, find_offdiag_idx))
  end

  val *= rho/2
  if compute_grad
    scale!(g, rho)
  end
 
  #= println("Val = ", val) =#
  #= println("g = ", g) =#
  return val
end

type ADMMData
  quic_data
  constr_data
  low_rank
  rho
  tol_abs
  tol_rel
  dual_balancing
  bb # Barzilai-Borwein stepsize
  tighten # Tighten penalty parameter
  mu
  tau
  B::Array{Float64,2}
  theta_prime::Array{Float64,2}
  #= theta::Array{Float64,2} =#
  Ainv2s::Array{Array{Float64,2},1}
  B0::Array{Float64,2}
  duals::Array{Array{Float64,2},1}
  T::Int64 # How often to run BB update
  eps_cor::Float64 # Safeguarding threshold for BB update
end

function ADMMData(emp_data, constr_data, lambda)
  data = ADMMData(QuadraticPenaltyData(emp_data.p), # quic_data
                  constr_data, # constr_data
                  false, # low_rank
                  1.0, # rho
                  1e-2, # tol_abs
                  1e-2, # tol_rel
                  true, # dual_balancing
                  false, # bb
                  false, # tighten
                  5.0, # mu
                  1.5, # tau
                  zeros(emp_data.p, emp_data.p), # B
                  zeros(emp_data.p, emp_data.p), # theta_prime
                  #= zeros(emp_data.p, emp_data.p), # theta =#
                  [zeros(emp_data.p, emp_data.p) for e = 1:emp_data.E], # Ainv2s
                  zeros(p, p), # B0 
                  [zeros(emp_data.p, emp_data.p) for e = 1:emp_data.E], # duals
                  3, # T
                  0.2, # eps_cor
                 )
  data.quic_data.lambda = lambda
  return data
end

function compute_Ainv2s(emp_data, admm_data, vars; low_rank = false)
  constr_data = admm_data.constr_data
  E = emp_data.E
  data = constr_data
  data.initialize_array = length(data.Breds) == 0
  Us_ind = emp_data.Us_ind
  Js_ind = emp_data.Js_ind

  fill!(data.B, 0.0)
  vec2mat(vars[end], emp_data, inplace = data.B)

  p = emp_data.p
  if low_rank
    # Assemble (I - B)
    fill!(data.Ainv, 0.0)
    for i = 1:p
      data.Ainv[i,i] = 1.0
    end

    #= data.Ainv[:] .-= data.B[:] =#
    BLAS.axpy!(-1.0, data.B, data.Ainv)
    #= BLAS.gemm!('T', 'N', 1.0, data.Ainv, data.Ainv, 0.0, data.Ainv2) =#
    BLAS.syrk!('U', 'T', 1.0, data.Ainv, 0.0, data.Ainv2)
    LinAlg.copytri!(data.Ainv2, 'U')
  end

  for e = 1:E
    fill!(data.diff, 0.0)
    if low_rank
      if data.initialize_array
        push!(data.Breds, zeros(length(Js_ind[e]), p))
        push!(data.Bmuldiff, zeros(length(Js_ind[e]), p))
        push!(data.Bredst, zeros(p, length(Js_ind[e])))
      end
      copy!(data.Breds[e], view(data.B, Js_ind[e], :))
      transpose!(data.Bredst[e], data.Breds[e])
      #= copy!(data.diff, data.Ainv2) =#
      BLAS.syrk!('U', 'T', -1.0, data.Breds[e], 0.0, data.diff)
      LinAlg.copytri!(data.diff, 'U')
      BLAS.axpy!(1.0, data.Breds[e], view(data.diff, Js_ind[e], :))
      BLAS.axpy!(1.0, data.Bredst[e], view(data.diff, :, Js_ind[e]))
      BLAS.axpy!(1.0, data.Ainv2, data.diff)
      # Symmetrize
      symmetrize!(data.diff)
      #= println("Symmetry diff: ", vecnorm(data.diff - data.diff')) =#
      #= println(size(data.Bredst[e]), ", ", size(data.Breds[e])) =#
      #= BLAS.gemm!('N', 'N', -1.0, data.Bredst[e], data.Breds[e], 1.0, data.diff) =#
      copy!(admm_data.Ainv2s[e], data.diff)
    else
      fill!(data.Ainv, 0.0)
      for i = 1:p
        data.Ainv[i, i] = 1
      end
      BLAS.axpy!(-1.0, view(data.B, Us_ind[e], :), view(data.Ainv, Us_ind[e], :))
      #= BLAS.gemm!('T', 'N', 1.0, data.Ainv, data.Ainv, 0.0, data.diff) =#
      BLAS.syrk!('U', 'T', 1.0, data.Ainv, 0.0, data.diff)
      LinAlg.copytri!(data.diff, 'U')
      copy!(admm_data.Ainv2s[e], data.diff)
    end
  end
  data.initialize_array = false
end

function min_admm(emp_data, admm_data)

  # Shorthands for variables
  E = emp_data.E
  p = emp_data.p
  sigmas_emp = emp_data.sigmas_emp
  B = admm_data.B
  theta_prime = admm_data.theta_prime
  I = emp_data.I
  Us = emp_data.Us
  theta_prime = admm_data.theta_prime
  qu_data = admm_data.quic_data
  low_rank = admm_data.low_rank
  B0 = admm_data.B0
  duals = admm_data.duals

  function solve_ml_quic!(vars, duals, rho, sigmas, e)
    #= println("Running quic for ", e) =#
    vec2mat(vars[end], emp_data, inplace = B)
    copy!(theta_prime, admm_data.Ainv2s[e])
    BLAS.axpy!(-1/rho, duals[e], theta_prime)
    #= println("dual = ", duals[e]) =#
    qu_data.theta0 = copy(vars[e])
    #= println("Theta_prime symmetric: ", vecnorm(theta_prime - theta_prime')) =#
    #= theta_prime2 = (I - Us[e]*B)'*(I - Us[e]*B) - duals[e]/rho =#
    #= println("Prime diff: ", vecnorm(theta_prime - theta_prime2)) =#
    #= var_old = copy(vars[e]) =#
    copy!(vars[e], quic(emp_data, qu_data, sigmas[e],
                        theta_prime = theta_prime, rho = rho))
    #= println("Diff: ", vecnorm(var_old - vars[e])) =#
    #= copy!(vars[e], theta) =#
  end

  #= constr_data = ConstraintData(p) =#
  constr_data = admm_data.constr_data

  function solve_constr!(vars, duals, rho)
    function lbfgs_obj(x, g)
      ret = constr_least_sq(x, g, vars, duals, rho, emp_data, constr_data, emp_data.Us_ind, emp_data.Js_ind, low_rank = low_rank)
      return ret
    end
    #= Profile.clear() =#
    #= @profile (minf, minx, ret) = Liblbfgs.lbfgs(vars[end], lbfgs_obj, print_stats = false) =#
    #= var_old = copy(vars[end]) =#
    (minf, minx, t, c, status) = lbfgsb(lbfgs_obj, vars[end], iprint=-1, pgtol = 1e-5, factr = 1e7)
    #= println(status) =#
    # Threshold
    Bloc = vars[end]
    Bloc[abs(Bloc) .< 1e-6] = 0.0

    #= copy!(view(vars[end],:), minx) =#
    #= println(var_old) =#
    #= println(minx) =#
    #= vars[end][:] = copy(minx) =#
    #= println("Diff: ", vecnorm(var_old-vars[end])) =#
  end

  #= function dual_update(emp_data, vars, e) =#
  #=   B = vec2mat(vars[end], emp_data) =#
  #=   Ainv = I - emp_data.Us[e] * B =#
  #=   return vars[e] - Ainv' * Ainv =#
  #= end =#

  function dual_update(admm_data, vars, primal_residuals; compute_B_only = false)
    compute_Ainv2s(emp_data, admm_data, vars, low_rank = low_rank)
    #= Ainv2s_comp = [] =#
    #= B = vec2mat(vars[end], emp_data) =#
    #= for e = 1:E =#
    #=   Ainv = I - emp_data.Us[e] * B =#
    #=   push!(Ainv2s_comp, vars[e] - Ainv'*Ainv) =#
    #= end =#
    if !compute_B_only
      for e = 1:E
        primal_residuals[e][:] = vars[e] - admm_data.Ainv2s[e]
      end
    end
    #= println("Diff: ", vecnorm(Ainv2s_comp - primal_residuals)) =#
  end

  function compute_bmap(admm_data, vars, rho, bmap)
    for e = 1:E
      copy!(view(bmap, 1 + (e-1)*p^2:e*p^2), admm_data.Ainv2s[e])
    end
  end

  solvers = []
  compute_grads_h = []
  compute_grads_g = []
  for e = 1:E
    push!(solvers, (vars, duals, rho) -> solve_ml_quic!(vars, duals, rho, sigmas_emp, e))
    #= push!(compute_grads_h, (g, vars) -> compute_grad_h!(g, vars, e)) =#
    #= push!(compute_grads_g, (g, vars) -> compute_grad_g!(g, vars, e)) =#
  end

  vars = []
  #= duals = [] =#
  chol = zeros(p, p)
  chol2 = zeros(p, p)
  for e = 1:E
    if emp_data.n > 2 * p
      # Inverses
      copy!(chol, sigmas_emp[e])
      LAPACK.potrf!('U', chol)
      LAPACK.potri!('U', chol)
      triu!(chol)
      transpose!(chol2, chol)
      tril!(chol2, 1)
      BLAS.axpy!(1.0, chol2, chol)
      push!(vars, copy(chol))
    else
      push!(vars, eye(p))
    end

    #= dual = 0.1*randn(p, p) =#
    #= dual = zeros(p, p) =#
    #= dual = (dual + dual')/2 =#
    #= push!(duals, dual) =#
  end
  #= B0 = 100 * triu(randn(p,p), 1) =#
  push!(vars, mat2vec(B0, emp_data))
  #= push!(vars, zeros(p*(p-1))) =#

  push!(solvers, solve_constr!)

  vars_result = admm(emp_data, admm_data, solvers, dual_update, vars, compute_bmap = compute_bmap)
  B_admm = vec2mat(vars_result[end], emp_data)
  #= println(vecnorm(vars_result[end] - mat2vec(B))) =#
  return B_admm
end

function min_admm_oracle(pop_data, emp_data, admm_data, lambdas)
  errors = zeros(length(lambdas))
  Bs = []
  B_admm = copy(admm_data.B0)
  for i = 1:length(lambdas)
    println("lambda = ", lambdas[i])
    admm_data.B0 = copy(B_admm)
    admm_data.quic_data.lambda = lambdas[i]
    B_admm = copy(min_admm(emp_data, admm_data))
    push!(Bs, B_admm)
    errors[i] = vecnorm(B_admm - pop_data.B)
  end

  (err, ind) = findmin(errors)
  return (Bs[ind], err, lambdas[ind], errors)
end

function min_constr_lh_oracle(pop_data, emp_data, lh_data, lambdas)
  errors = zeros(length(lambdas))
  Bs = []
  B_lh = copy(lh_data.B0)
  for i = 1:length(lambdas)
    lh_data.lambda = lambdas[i]
    lh_data.B0 = copy(B_lh)
    B_lh = copy(min_constraint_lh(emp_data, lh_data))
    push!(Bs, B_lh)
    errors[i] = vecnorm(B_lh - pop_data.B)
  end

  (err, ind) = findmin(errors)
  return (Bs[ind], err, lambdas[ind], errors)
end

function combined_oracle(pop_data, emp_data, admm_data, lh_data, lambdas)
  (B1, err1, lambda1, errors1) = min_admm_oracle(pop_data, emp_data, admm_data, lambdas)
  lh_data.x_base = mat2vec(B1, emp_data)
  lh_data.upper_bound = vecnorm(pop_data.B - B1)^2
  lh_data.B0 = copy(B1)
  (B2, err2, lambda2, errors2) = min_constr_lh_oracle(pop_data, emp_data, lh_data, lambdas) 
  return (B1, B2, err1, err2, lambda1, lambda2, errors1, errors2)
end

# LLC functionality
function delete_shift_ind(l, i)
  return union(l[l .< i], l[l .> i] - 1)
end

function llc(pop_data, emp_data, lambdas)
  # Shortcuts
  p = emp_data.p
  E = emp_data.E
  Us_ind = emp_data.Us_ind
  Js_ind = emp_data.Js_ind
  sigmas = emp_data.sigmas_emp

  Bs = [zeros(p,p) for i in 1:length(lambdas)]
  #= B = zeros(p, p) =#

  for u = 1:p
    T::Array{Float64,2} = fill(0.0, 0, p-1)
    t::Array{Float64,1} = fill(0.0, 0)
    for e = 1:E
      if u in Us_ind[e]
        for i in Js_ind[e]
          row = zeros(p-1)
          target_elem_inds = delete_shift_ind(Us_ind[e], u)
          source_elem_inds = setdiff(Us_ind[e], u)
          copy!(view(row, target_elem_inds), view(sigmas[e], i, source_elem_inds))
          #= copy!(row[target_elem_inds], view(sigmas[e], i, source_elem_inds)) =#
          row[i <= u ? i : i - 1] = 1.0
          T = vcat(T, row')
          t = vcat(t, sigmas[e][i, u])
        end
      end
    end
    #= b = T \ t =#
    path = fit(LassoPath, T, t, λ = lambdas)
    all_but_u = setdiff(1:p, u)
    for ind in 1:length(lambdas)
      copy!(view(Bs[ind], u, all_but_u), view(path.coefs, :, ind))
    end
    #= println(size(repmat(pop_data.B[u, all_but_u]', size(path.coefs, 2), 1))) =#
    #= (err, ind) = findmin(sum((path.coefs - repmat(pop_data.B[u, all_but_u]', size(path.coefs, 1), 2)).^2)) =#
    #= (err, ind) = findmin(sum(broadcast(-, path.coefs, reshape(pop_data.B[u, all_but_u], p-1, 1)).^2, 1)) =#
    #= copy!(view(B, u, all_but_u), b) =#
    #= copy!(view(B, u, all_but_u), view(path.coefs, :,ind)) =#
  end

  #= (err, ind) = findmin(sum(broadcast(-, path.coefs, reshape(pop_data.B[u, all_but_u], p-1, 1)).^2, 1)) =#
  errors = [vecnorm(B - pop_data.B) for B in Bs]
  (err, ind) = findmin(errors)
  #= println(path.λ[ind]) =#

  return (Bs[ind], err, lambdas[ind], errors)
end

end
