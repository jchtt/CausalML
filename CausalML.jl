# vim: ts=2 sw=2 et
module CausalML
export gen_b, gen_overlap_cycles, gen_clusters, PopulationData, EmpiricalData, lh, l1_wrap, mat2vec, vec2mat, min_vanilla_lh, VanillaLHData, min_constraint_lh, quic_old, quic, QuadraticPenaltyData, min_admm, ConstraintData, ADMMData, QuadraticPenaltyData_old, min_admm_old, ConstraintData_old, ADMMData_old, min_admm_oracle, llc, symmetrize!, min_constr_lh_oracle, combined_oracle, savevar, k_fold_split, min_admm_cv, min_constr_lh_cv, combined_cv

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
using LightGraphs
#= using Convex =#
#= using SCS =#

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

function gen_overlap_cycles(p, horz, vert, std)
  if !(p == 2*vert + 2*horz || mod(p - 2*vert - 2*horz, 2*horz + vert - 1) == 0)
    error("Inadmissible choice of p, vert and horz.")
  end

  B = zeros(p, p)
  element() = std*(2*rand()-1)
  #= element() = 1 =#

  # Draw first cycle
  for i = 1:horz
    B[i, i+1] = element()
  end
  top_right = horz+1
  counter = top_right
  for i = 0:vert-1
    B[counter + i, counter + i + 1] = element()
  end
  counter += vert
  bottom_right = counter
  for i = 0:horz-1
    B[counter + i, counter + i + 1] = element()
  end
  counter += horz
  for i = 0:vert-2
    B[counter + i, counter + i + 1] = element()
  end
  counter += vert-1
  B[counter, 1] = element()
  counter += 1

  # Draw remaining cycles
  remaining = p - 2*vert - 2*horz
  while remaining > 0
    B[top_right, counter] = element()
    for i = 0:horz-2
      B[counter + i, counter + i + 1] = element()
    end
    counter += horz - 1
    top_right = counter
    #= println("Top right: ", top_right) =#
    for i = 0:vert-1
      B[counter + i, counter + i + 1] = element()
    end
    counter += vert
    bottom_right_new = counter
    #= println("Bottom right: ", bottom_right_new) =#
    for i = 0:horz-2
      B[counter + i, counter + i + 1] = element()
    end
    counter += horz-1
    B[counter, bottom_right] = element()
    bottom_right = bottom_right_new
    counter += 1

    remaining -= 2*horz + vert - 1
  end
  
  return B
end

function gen_clusters(p, d, std)
  cluster_size = d + 1
  num_clusters = ceil(Int64, p/cluster_size)
  B = zeros(p, p)
  for i = 1:num_clusters
    actual_size = min(cluster_size, p - (i-1)*cluster_size)
    A = std*2*(rand(actual_size, actual_size) - 1)
    for j = 1:actual_size
      A[j, j] = 0
    end
    r = ((i-1)*cluster_size + 1):min(p, i*cluster_size)
    B[r, r] = A
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

  function PopulationData(p, d, std, experiment_type; k = 1, graph_type = "random", horz = 1, vert = 1)
    ret = new()
    ret.p = p
    ret.d = d
    ret.std = std
    #= ret.B = gen_b(p, d, std/sqrt(d*p)) =#
    if graph_type == "random"
      ret.B = gen_b(p, d, std/d)
    elseif graph_type == "random_no_norm"
      ret.B = gen_b(p, d, std)
    elseif graph_type == "overlap_cycles"
      ret.d = 3
      ret.B = gen_overlap_cycles(p, horz, vert, std/ret.d)
    elseif graph_type == "clusters"
      ret.B = gen_clusters(p, d, std/d)
    else
      error("Unknown graph type")
    end
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
    elseif experiment_type == "bounded"
      # Check condition
      if p <= k^2/2
        error("Cannot contsruct bounded experiments because k is too large")
      end

      # Build graph
      num_cliques = floor(Int64, 2*p/(k*(k+1)))
      rest = p - num_cliques * (k*(k+1))/2
      l = ceil(Int64, sqrt(0.25+2*rest)-0.5) + 1
      m = num_cliques * (k+1) + l
      #= m = ceil(Int64, 2*p/k)+1 =#
      L = zeros(Int64, m, m) # adjacency matrix

      edges_missing = p
      for i = 1:m
        #= println("Edges missing: ", edges_missing) =#
        #= println(L) =#
        if edges_missing == 0
          break
        end
        prev_connections = sum(L[1:i-1, i])
        new_connections = k - prev_connections
        actual_new_connections = min(new_connections, edges_missing, m-i)
        #= inds = (i+1:p)[randperm(p-i)][1:actual_new_connections] =#
        #= inds = i+1+1:i+actual_new_connections =#
        for j = i+1:i+actual_new_connections
        #= for j in inds =#
          L[j, i] = 1
          L[i, j] = 1
        end
        edges_missing -= actual_new_connections
      end
      println("Edges missing: ", edges_missing)
      #= println(L) =#

      # Extract experiments
      for i = 1:m
        push!(ret.Js_ind, [])
      end
      #= println(ret.Js_ind) =#
      for (node, (i, j)) in enumerate(zip(findn(triu(L))...))
        push!(ret.Js_ind[i], node)
        push!(ret.Js_ind[j], node)
      end

      for i = 1:m
        mask = trues(p)
        mask[ret.Js_ind[i]] = false
        push!(ret.Us_ind, find(mask))

        U = zeros(p)
        U[mask] = 1
        U = diagm(U)

        push!(ret.Us, U)
      end
    elseif experiment_type == "all_but_one"
      for e = 1:p
        mask = trues(p)
        mask[e] = false
        push!(ret.Us_ind, find(~mask))
        push!(ret.Js_ind, find(mask))
        # println(~(mask.==1))

        U = ones(p)
        U[mask] = 0
        U = diagm(U)

        push!(ret.Us, U)
      end
    else
      error("Unrecognized experiment_type")
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

  function EmpiricalData(pop_data, n; store_samples = false, Xs = [])
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
    ret.Xs = Xs
    calc_new_samples = length(Xs) == 0

    for e = 1:ret.E
      # Population level matrices
      Ainv = pop_data.Ainvs[e]

      # Empirical covariance
      if calc_new_samples
        Z = randn(ret.p, ret.n)
        X = Ainv \ Z
      else
        X = ret.Xs[e]
      end
      if store_samples
        push!(ret.Xs, copy(X))
      end
      sigma_emp = X*X'/n
      push!(ret.sigmas_emp, sigma_emp)
    end

    return ret
  end
end

function k_fold_split(pop_data, emp_data, k, i)
  """
  Return two new emp_data objects, one with the training data, the other with the testing data from the ith fold of a k-fold split.
  """

  n = emp_data.n
  fold_length = ceil(Int64, n/k)

  if i < k
    actual_length = fold_length
  elseif i == k
    actual_length = n - (k - 1) * fold_length
  end

  test_r = (i-1)*fold_length + 1:min(i*fold_length, n)
  train_r = setdiff(1:n, test_r)

  emp_data_test = EmpiricalData(pop_data, length(test_r), store_samples = false, Xs = [X[:, test_r] for X in emp_data.Xs])
  emp_data_train = EmpiricalData(pop_data, length(train_r), store_samples = false, Xs = [X[:, train_r] for X in emp_data.Xs])

  return (emp_data_train, emp_data_test)
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
  Ainv::Array{Float64,2} # I - B
  Ainve::Array{Float64,2} # I - U*B
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
  use_constraint::Bool # Should we constraint to a ball?
  continuation::Bool # Should we use continuation in oracle
end

function VanillaLHData(p, lambda, B0)
  return VanillaLHData(lambda, B0,
                       zeros(p, p), zeros(p, p),
                       zeros(p, p),
                       zeros(p, p), # Ainve
                       zeros(p, p),
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
                       true, # use_constraint
                       true, # continuation
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

  temp_no_low_rank = false

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


      copy!(data.CholfacteFactor, data.CholfactFactor)
      data.Cholfacte = LinAlg.Cholesky(data.CholfacteFactor, :U)

      try
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
        val += BLAS.dot(p^2, data.Ainv2e, 1, emp_data.sigmas_emp[e], 1)
      catch err
        if isa(err, Base.LinAlg.PosDefException)
          println("Warning: cannot use low rank decomposition because matrix is not positive definite any more.")
          temp_no_low_rank = true
        else
          rethrow()
        end
      end
    end
    if !low_rank || temp_no_low_rank
      # Compute theta
      fill!(data.Ainve, 0.0)
      for i = 1:p
        data.Ainve[i, i] = 1
      end
      BLAS.axpy!(-1.0, view(data.B, emp_data.Us_ind[e], :), view(data.Ainve, emp_data.Us_ind[e], :))
      BLAS.gemm!('T', 'N', 1.0, data.Ainve, data.Ainve, 0.0, data.Ainv2e)
      symmetrize!(data.Ainv2e)
      val += BLAS.dot(p^2, emp_data.sigmas_emp[e], 1, data.Ainv2e, 1)
      copy!(data.CholfacteFactor, data.Ainv2e)
      # Use Symmetric() to handle round-off errors above
      try
        cholfact!(Symmetric(data.CholfacteFactor))
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
        val -= sum(2*log(diag(data.CholfacteFactor)))
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

      temp_no_low_rank = false
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
        LAPACK.potri!('U', data.CholfacteFactor)
        data.CholfacteFactor[emp_data.strict_lower_idx] = 0.0
        transpose!(data.CholfacteFactorT, data.CholfacteFactor)
        data.CholfacteFactor[emp_data.diag_idx] = 0.0
        BLAS.axpy!(1.0, data.CholfacteFactorT, data.CholfacteFactor)
        BLAS.axpy!(-1.0, data.CholfacteFactor, data.diff)
        BLAS.gemm!('N', 'N', 2.0, data.Ainve, data.diff, 0.0, data.Gmat)
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


function min_vanilla_lh(emp_data, lh_data)
  println()
  println("Solving vanilla lh")
  dim = 2*emp_data.p*(emp_data.p-1)
  p = emp_data.p
  low_rank = lh_data.low_rank

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
        #= inner_converged = true =#
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
  inner_min_iterations::Int64 # How many iterations to start with
  inner_max_iterations::Int64 # How many iterations to do at most in the inner loop
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
  Ut::Array{Float64,2}
  G::Array{Float64,2}
  Gmin::Array{Float64,2}
  theta_new::Array{Float64,2}
  diff::Array{Float64,2}
  uvec::Array{Float64,1}
  hard_thresh::Float64 # Cut-off for each major iteration
  max_iterations::Int64 # maximum outer iterations
end

function QuadraticPenaltyData(p, lambda, inner_mult, inner_min_iterations, inner_max_iterations,
                              theta0,
                             tol, inner_tol, relaxation, beta, eps,
                             print_stats, hard_thresh, max_iterations
                            )
  return QuadraticPenaltyData(
                              # For quic
                              lambda,
                              inner_mult,
                              inner_min_iterations,
                              inner_max_iterations,
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
                              zeros(p, p), # Ut
                              zeros(p, p), # G
                              zeros(p, p), # Gmin
                              zeros(p, p), # theta_new
                              zeros(p, p), # diff
                              zeros(p), # uvec
                              hard_thresh,
                              max_iterations,
                             )
end

function QuadraticPenaltyData(p)
  return QuadraticPenaltyData(p,
                              1e-3, # lambda
                              1/3, # inner_mult
                              2, # inner_min_iterations
                              1000, # inner_max_iterations
                              eye(p), # theta0
                              1e-4, # tol
                              1e-8, # inner_tol
                              0.2, # relaxation
                              0.9, # beta
                              0.01, # eps
                              false, # print_stats
                              1e-10, # hard_thresh
                              50, # max_iterations
                             )
end

function quic_cg(theta0, # base matrix
                 G, # gradient
                 Z, # orthant indicator
                 lambda,
                 rho; # regularization parameter
                 k_max = 10,
                 tol = 1e-8
                )

  p = size(theta0, 1)
  inactive_inds = (theta0 .!= 0) & (abs.(G) .<= lambda)
  W = inv(theta0)
  symmetrize!(W)
  theta = zeros(p, p)
  theta_vec = reshape(theta, p^2) 
  R = -(G + lambda * Z)
  R[inactive_inds] = 0
  k = 0
  r = reshape(R, p^2)
  q = copy(r)

  while k <= min(k_max, p^2) && vecnorm(r) > tol
    Q = reshape(q, p, p)
    Y = W * Q * W + rho * Q
    Y[inactive_inds] = 0
    y = reshape(Y, p^2)
    alpha = sum(r.^2)/sum(q .* y)
    theta_vec += alpha * q
    r_new = r - alpha * y
    beta = sum(r_new.^2)/sum(r.^2)
    r = r_new
    q = r + beta * q
    k += 1
  end

  return reshape(theta_vec, p, p)
end

#= function quic_fista(theta, =#
#=                     c # regularization =#
#=                    ) =#
#=   t_old = 1 =#
#=   t = (1 + sqrt(1 + 4 t^2))/2 =#
#=   theta_old = copy(theta) =#
#=   theta_hat = theta + (t-1)/ =#
#=   for t = 1:t_max =#
#=     soft_thresh.(theta_hat - G/c - W * (theta_hat - theta) * W/c - rho * ones(p, p) / c, lambda/c) =#
#=   end =#
#= end =#

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
  inner_min_iterations = data.inner_min_iterations
  inner_max_iterations = data.inner_max_iterations
  theta0 = data.theta0
  tol = data.tol
  inner_tol = data.inner_tol
  relaxation = data.relaxation
  beta = data.beta
  eps = data.eps
  print_stats = data.print_stats
  status = 0


  (p1, p2) = size(sigma)
  if p1 != p2
    throw(ArgumentError("sigma needs to be a square matrix"))
  end
  p = p1

  diag_idx = diagm(trues(p,1))
  offdiag_idx = ~diag_idx
  find_offdiag_idx = find(~diag_idx)

  converged_outer = false
  counter = 1
  alpha = 1.0
  copy!(data.W, theta0)
  LAPACK.potrf!('U', data.W)

  #= theta = copy(theta0) =#
  copy!(data.theta, theta0)
  symmetrize!(data.theta)
  #= println("Theta0 = ", theta0) =#
  #= println("sigma = ", sigma) =#
  #= println("theta_prime = ", theta_prime) =#
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
  #= f_old = BLAS.dot(p^2, data.theta, 1, sigma, 1) - sum(2*log(diag(data.W))) + rho/2 * vecnorm(data.diff)^2 + lambda * vecnorm(data.theta, 1) =#
  f_old = BLAS.dot(p^2, data.theta, 1, sigma, 1) - sum(2*log(diag(data.W))) + rho/2 * vecnorm(data.diff)^2 + lambda * vecnorm(data.theta[offdiag_idx], 1)

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
  while ~converged_outer && counter <= data.max_iterations
    if print_stats
      println("Outer iteration: ", counter)
      println("cond(theta) = ", cond(data.theta))
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

    #= S = findn(triu((abs(data.G) .>= lambda - eps) | (data.theta .!= 0))) =#
    S = findn(triu(trues(p, p)))
    #= println(triu((abs(data.G) .>= lambda - eps) | (data.theta .!= 0))) =#

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
    inner_iterations = floor(Int64, inner_min_iterations + counter * inner_mult)
    inner_converged = false
    #= while r <= inner_iterations && ~inner_converged #|| ~descent_step =#
    #= active_inds = shuffle(collect(zip(S...))) =#
    active_inds = zip(S...)

    #= # DEBUG START =#
    #= println("W accuracy: ", vecnorm(eye(p) - data.W * data.theta)) =#
    #= # DEBUG END =#

    #= while r <= inner_max_iterations && (r <= inner_iterations || ~inner_converged) #|| ~descent_step =#
    #= while ~inner_converged #|| ~descent_step =#
    while r <= inner_max_iterations && ~inner_converged #|| ~descent_step
    #= while r <= 100 =#
      #= if mod(r, 1) == 0 =#
        #= if print_stats =#
        #=   println("Inner iteration ", r) =#
        #= end =#

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

      #= for (i, j) in zip(S...) =#
      for (i, j) in active_inds
        a = data.W[i,j]^2
        a += rho
        if i != j
          a += data.W[i,i]*data.W[j,j]
        end
        b = 0.0
        @simd for k = 1:p
          @inbounds b += data.W[k,i] * data.U[j,k]
        end
        #= b += sum(data.W[:,i] .* data.U[j, :]) =#

        #= if i == 1 && j == 1 =#
        #=   println(b) =#
        #= end =#
        #= b += sum(W[:,i] .* U[j,:]) =#
        #= copy!(data.uvec, view(data.U, j, :)) =#
        #= b += BLAS.dot(view(data.W, :, i), data.uvec) =#
        b += sigma[i,j]
        b -= data.W[i,j]
        #= b -= data.W[j,i] =#
        #= if i == 1 && j == 1 =#
        #=   println(b) =#
        #= end =#
        b += rho * (data.theta[i,j] + data.D[i,j] - data.theta_prime[i,j])
        #= if i == 1 && j == 1 =#
        #=   println(b) =#
        #= end =#
        c = data.theta[i,j] + data.D[i,j]

        #= # DEBUG start =#
        #= if i == 1 =#
        #=   mu = 1 =#
        #=   f0 = sum((sigma - data.W).*data.D) + 0.5*trace(data.W*data.D*data.W*data.D) + rho/2*vecnorm(data.theta + data.D - data.theta_prime)^2 =#
        #=   data.D[i,j] += mu =#
        #=   if i != j =#
        #=     data.D[j,i] += mu =#
        #=   end =#
        #=   f_plus = sum((sigma - data.W).*data.D) + 0.5*trace(data.W*data.D*data.W*data.D) + rho/2*vecnorm(data.theta + data.D - data.theta_prime)^2 =#
        #=   data.D[i,j] -= 2*mu =#
        #=   if i != j =#
        #=     data.D[j,i] -= 2*mu =#
        #=   end =#
        #=   f_min = sum((sigma - data.W).*data.D) + 0.5*trace(data.W*data.D*data.W*data.D) + rho/2*vecnorm(data.theta + data.D - data.theta_prime)^2 =#
        #=   #1= c_ideal = f0 =1# =#
        #=   b_ideal = (f_plus - f_min)/(2*mu) =#
        #=   a_ideal = 2*(f_plus - mu*b_ideal - f0)/(mu^2) =#
        #=   if i != j =#
        #=     b_ideal /= 2 =#
        #=     a_ideal /= 2 =#
        #=   end =#
        #=   #1= if i == j =1# =#
        #=     if counter == 1 =#
        #=       println("-------------------------------------------") =#
        #=       println("Theory: a = ", a_ideal, ", b = ", b_ideal) =#
        #=       println("Practice: a = ", a, ", b = ", b) =#
        #=     end =#
        #=   #1= end =1# =#
        #=   data.D[i,j] += mu =#
        #=   if i != j =#
        #=     data.D[j,i] += mu =#
        #=   end =#
        #= end =#
        #= # DEBUG end =#

        #= if i == j =#
        #=   mu = -b/a =#
        #= else =#
          mu = -c + soft_thresh(c - b/a, lambda/a)
        #= end =#

        #= if i == p && j == p =#
        #=   println("j = ", j, ", ", "a = ", a, ", b = ", b, ", c = ", c, ", b/a = ", b/a) =#
        #=   println("sigma diff: ", sigma[i, j] - data.W[i, j], ", rho diff: ", rho * (data.theta[i,j] + data.D[i,j] - data.theta_prime[i,j]), ", quadratic: ", sum(data.W[:,i] .* data.U[j, :]), ", quadratic leading: ", data.W[j, i] * data.U[j, i] =#
        #=          ) =#
        #= #1=   println(soft_thresh(c - b/a, lambda/a)) =1# =#
        #= #1=   println(mu) =1# =#
        #= #1= #2=   println(data.D[i,j]) =2# =1# =#
        #= #1= #2=   println(data.D[i,j] + mu) =2# =1# =#
        #= end =#

        if mu != 0.0
          data.D[i,j] += mu
          if i != j
            data.D[j,i] += mu
          end
          #= data.U[:,i] += mu * data.W[:,j] =#
          @simd for k = 1:p
            @inbounds data.U[k,i] += mu * data.W[k,j]
          end
          #= BLAS.axpy!(mu, view(data.W, :, j), view(data.U, :, i)) =#
          #= BLAS.axpy!(mu, data.W[:, j], data.U[:,i]) =#
          if i != j
            #= data.U[:,j] += mu * data.W[:,i] =#
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

      #= data.diff = data.theta + data.D =#
      comparison = relaxation * (sum(data.G .* data.D) + lambda * (vecnorm(data.diff[offdiag_idx], 1) - vecnorm(data.theta[offdiag_idx], 1)))
      #= comparison = relaxation * (sum(data.G .* data.D) + lambda * (vecnorm(data.theta + data.D, 1) - vecnorm(data.theta, 1))) =#
      descent_step = comparison < 0
      #= if print_stats =#
      #=   println("descent = ", descent_step) =#
      #= end =#
      #= diff = vecnorm(data.D - data.D_old) =#
      copy!(data.diff, data.D)
      BLAS.axpy!(-1.0, data.D_old, data.diff)
      cur_norm = vecnorm(data.D)
      diff = vecnorm(data.diff)/max(cur_norm, 1e-4)
      #= println("Inner difference: ", diff, ", ", vecnorm(data.diff)) =#
      if diff < inner_tol
        inner_converged = true
        #= if print_stats =#
        #=   println("Inner converged, |D| = ", vecnorm(data.D), ", comparison = ", comparison) =#
        #= end =#
      end
      r += 1
    end

    #= # DEBUG START =#
    #= # Compute cg solution =#
    #= Z = zeros(p, p) =#
    #= Z[data.theta .> 0] = 1 =#
    #= Z[data.theta .< 0] = -1 =#
    #= Z[(data.theta .== 0) & (data.G .> lambda)] = -1 =#
    #= Z[(data.theta .== 0) & (data.G .< lambda)] = 1 =#
    #= Z[(data.theta .== 0) & (abs(data.G) .<= lambda)] = -data.G[abs(data.G) .<= lambda]/lambda =#
    #= actives = findn((data.theta .== 0) & (abs(data.G) .<= lambda)) =#
    #= inactives = findn((data.theta .!= 0) | (abs(data.G) .> lambda)) =#

    #= D_cg = quic_cg(data.theta, data.G, Z, lambda, rho) =#
    #= proj_step = data.theta + D_cg =#
    #= proj_step[sign(proj_step) .!= sign(Z)] = 0 =#

    #= # Compare to solution found by convex solver =#
    #= println("Inner iterations: ", r) =#
    #= Dvar = Variable(p, p) =#
    #= theta_inv = inv(data.theta) =#
    #= symmetrize!(theta_inv) =#
    #= theta_inv_sqrt = sqrtm(theta_inv) =#
    #= symmetrize!(theta_inv_sqrt) =#
    #= problem = minimize(sum((sigma - theta_inv) .* Dvar) + 0.5*vecnorm(theta_inv_sqrt * Dvar * theta_inv_sqrt)^2 + rho / 2 * vecnorm(data.theta + Dvar - theta_prime)^2 + lambda * vecnorm(data.theta + Dvar - diagm(diag(data.theta + Dvar)), 1)) =#
    #= #1= problem = minimize(sum((sigma - theta_inv) .* Dvar) + rho / 2 * vecnorm(data.theta + Dvar - theta_prime)^2) =1# =#
    #= solve!(problem, SCSSolver(verbose = false, max_iters = 10000)) =#
    #= D_convex = Dvar.value =#
    #= println("Comparison = ", vecnorm(data.D - D_convex), ", ", vecnorm(D_cg - D_convex)) =#
    #= println("Relative: ", vecnorm(data.D - D_convex)/vecnorm(D_convex), ", ", vecnorm(D_cg - D_convex)/vecnorm(D_convex)) =#
    #= println("First coordinates: ", D_convex[1,1], ", ", data.D[1,1]) =#
    #= objective(Dvar) = sum((sigma - theta_inv) .* Dvar) + 0.5*trace(theta_inv * Dvar * theta_inv * Dvar) + rho / 2 * vecnorm(data.theta + Dvar - theta_prime)^2 + lambda * vecnorm(data.theta + Dvar, 1) =#
    #= objective2(Dvar) = sum((sigma - theta_inv) .* Dvar) + 0.5*vecnorm(theta_inv_sqrt * Dvar * theta_inv_sqrt)^2 + rho / 2 * vecnorm(data.theta + Dvar - theta_prime)^2 + lambda * vecnorm(data.theta + Dvar, 1) =#
    #= println("Objectives: ", objective(D_convex), ", ", objective(data.D)) =#
    #= println("Objectives (2): ", objective2(D_convex), ", ", objective2(data.D)) =#
    #= println("Min egis: ", eigmin(data.theta + D_convex), ", ", eigmin(data.theta + data.D), ", ", eigmin(proj_step)) =#
    #= println("Convex result: ", data.theta + D_convex) =#
    #= println("Iterative result: ", data.theta + data.D) =#
    #= println("CG result: ", proj_step) =#
    #= #1= if counter > 1 =1# =#
    #=   #1= data.D = D_convex =1# =#
    #= #1= end =1# =#
    #= #1= println(data.D) =1# =#
    #= #1= println(data.theta + data.D) =1# =#
    #= println("Accuracy: ", vecnorm(data.U - data.W' * data.D)) =#
    #= # DEBUG END =#

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

    if print_stats
      println("Inner iterations: ", r)
    end

    if r > inner_max_iterations
      status = 1
    end

    # Compute self-concordant step size
    #= eps_term = sqrt(trace(data.W * data.D * data.W * data.D) + rho * sum(data.D)^2) =#
    #= eps_term = sqrt(trace(data.W * data.D * data.W * data.D)) =#
    eps_term = 0.0
    transpose!(data.Ut, data.U)
    @simd for i = 1:p
      @simd for j = 1:p
        @inbounds eps_term += data.U[i, j] * data.Ut[i, j]
      end
    end
    eps_term = sqrt(eps_term)

    if eps_term > 3/40
      tau = 1/(eps_term + 1)
    else
      tau = 1
    end

    posdef = false

    f_new = 0.0

    while ~posdef
      copy!(data.theta_new, data.theta)
      BLAS.axpy!(tau, data.D, data.theta_new)
      copy!(data.W, data.theta_new)
      try
        LAPACK.potrf!('U', data.W)
        f_new = BLAS.dot(p^2, data.theta_new, 1, sigma, 1) - sum(2*log(diag(data.W))) + rho/2 * vecnorm(data.diff)^2 + lambda * vecnorm(data.theta_new, 1)
        posdef = true
      catch
        tau = beta * tau
        continue
      end
    end

    if print_stats
      println("tau = ", tau)
    end
#=     # COMMENT START =#
#=     # Compute Armijo step size =#
#=     alpha = 1.0 =#
#=     f_new = 0.0 =#
#=     while true =#
#=       #1= data.theta_new[:] = data.theta + alpha * data.D =1# =#
#=       copy!(data.theta_new, data.theta) =#
#=       BLAS.axpy!(alpha, data.D, data.theta_new) =#
#=       #1= W[:] = copy(theta_new) =1# =#
#=       copy!(data.W, data.theta_new) =#
#=       try =#
#=         LAPACK.potrf!('U', data.W) =#
#=         if any(diag(data.W) .<= 0.0) =#
#=           throw(DomainError()) =#
#=         end =#
#=       catch =#
#=         #1= alpha *= beta =1# =#
#=         #1= if print_stats =1# =#
#=         #1=   println("Warning, not psd!, ", minimum(diag(data.W))) =1# =# 
#=         #1= end =1# =#
#=         #1= continue =1# =#
#=       end =#
#=       min_eig = eigmin(data.theta_new) =#
#=       if min_eig < 1e-1 =#
#=         #1= println("Min eig: ", min_eig) =1# =#
#=         alpha *= beta =#
#=         continue =#
#=       end =#

#=       copy!(data.diff, data.theta_new) =#
#=       BLAS.axpy!(-1.0, data.theta_prime, data.diff) =#
#=       f_new = BLAS.dot(p^2, data.theta_new, 1, sigma, 1) - logdet(data.theta_new) + rho/2 * vecnorm(data.diff)^2 + lambda * vecnorm(data.theta_new[offdiag_idx], 1) =#
#=       #1= f_new = BLAS.dot(p^2, data.theta_new, 1, sigma, 1) - logdet(data.theta_new) + rho/2 * vecnorm(data.diff)^2 + lambda * vecnorm(data.theta_new, 1) =1# =#
#=       #1= f_new = BLAS.dot(p^2, data.theta_new, 1, sigma, 1) - sum(2*log(diag(data.W))) + rho/2 * vecnorm(data.diff)^2 + lambda * vecnorm(data.theta_new, 1) =1# =#
#=       #1= f_new = sum((data.theta_new) .* sigma) - sum(2*log(diag(data.W))) + rho/2*vecnorm(data.theta_new - data.theta_prime)^2 + lambda * vecnorm(data.theta_new, 1) =1# =#
#=       #1= f_new = sum((theta_new) .* sigma) - logdet(theta_new) + rho/2*vecnorm(theta_new - theta_prime)^2 + lambda * vecnorm(theta_new, 1) =1# =#
#=       #1= f_new = sum((theta_new) .* sigma) - logdet(theta_new) + vecnorm(theta_new, 1) =1# =#
#=       if f_new <= f_old + alpha * comparison =#
#=         break =#
#=       else =#
#=         alpha *= beta =#
#=         if print_stats =#
#=           #1= print("theta = ") =1# =#
#=           #1= println(data.theta) =1# =#
#=           #1= print("D = ") =1# =#
#=           #1= println(data.D) =1# =#
#=           #1= print("sigma = ") =1# =#
#=           #1= println(sigma) =1# =#
#=           #1= print("theta_prime = ") =1# =#
#=           #1= println(theta_prime) =1# =#
#=           #1= print("rho = ") =1# =#
#=           #1= println(rho) =1# =#
#=           #1= print("lambda = ") =1# =#
#=           #1= println(lambda) =1# =#
#=           println("decrease alpha = ", alpha, ", f_new = ", f_new, ", f_old = ", f_old) =#
#=         end =#
#=         #1= println("theta_new = ", theta_new) =1# =#
#=         #1= println("theta = ", theta) =1# =#
#=       end =#
#=     end =#
#=     # COMMENT END =#

    if print_stats
      println("alpha = ", alpha)
    end

    #= if vecnorm(theta_new - theta) < tol =#
    #=   converged_outer = true =#
    #= else =#
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
    #= outer_it_counter += 1 =#
    counter += 1
    if counter > data.max_iterations
      println("Warning: maximum iteration count for QUIC reached, stopping.")
    end
  end

  if length(g) > 0
    #= g[:] = copy(G) =#
    copy!(g, data.G)
  end

  return (data.theta, status)
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
    theta,_ = quic(emp_data, qu_data, sigmas[e],
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
  status = 0

  converged = false
  balanced = false
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
  r = 1
  while ~converged && r <= admm_data.max_iterations
    #= status = Dict() =#
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
    println("Norm difference: ", sqrt(sum([vecnorm(mat)^2 for mat in vars - vars_old])))
    println("Primal residual: ", res_norm)
    println("Dual residual: ", dual_res_norm)
    #= if vecnorm(vars - vars_old) < tol =#
    tol_primal = sqrt(dim_primal) * tol_abs + max(sqrt(sum([vecnorm(var)^2 for var in vars])), vecnorm(bmap)) * tol_rel
    tol_dual = sqrt(dim_dual) * tol_abs + sqrt(sum([vecnorm(dual)^2 for dual in duals])) * tol_rel
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
        balanced = true
        bal_counter = 1
      elseif dual_res_norm > mu * res_norm
        rho /= tau
        balanced = true
        bal_counter = 1
      end
    end

    # Tightening
    if tighten && counter >= 3
      if res_norm > res_norm_old && !balanced
        rho *= 1.5
      end
    end

    if balanced
      if bal_counter >= 3
        balanced = false
      end
      bal_counter += 1
    end

    res_norm_old = copy(res_norm)

    #= global admm_path =#
    #= status["vars"] = deepcopy(vars) =#
    #= status["duals"] = deepcopy(duals) =#
    #= status["bmap"] = deepcopy(bmap) =#
    #= status["rho"] = copy(rho) =#
    #= push!(admm_path[end], status) =#

    #= if (counter <= 40 || admm_data.quic_data.lambda > 0.005) && mod(counter, 20) == 0 =#
    #=   println("Serializing") =#
    #=   savevar(debugfname, admm_path) =#
    #= end =#

    println("rho = ", rho)
    #= println("B = ", vars[end]) =#
    vars_old = deepcopy(vars)
    r += 1
  end

  if r > admm_data.max_iterations
    status = 1
  end

  return (vars, status)
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
  max_iterations::Int64 # maximum number of iterations
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
                  2000, # max_iterations
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

    # DEBUG START
    qu_data.theta0 = copy(vars[e])
    #= qu_data.theta0 = eye(p) =#
    # DEBUG END

    #= println("Theta_prime symmetric: ", vecnorm(theta_prime - theta_prime')) =#
    #= theta_prime2 = (I - Us[e]*B)'*(I - Us[e]*B) - duals[e]/rho =#
    #= println("Prime diff: ", vecnorm(theta_prime - theta_prime2)) =#
    #= var_old = copy(vars[e]) =#
    #= println(theta_prime) =#
    #= println(rho) =#
    #= println(qu_data) =#
    copy!(vars[e], quic(emp_data, qu_data, sigmas[e],
                        theta_prime = theta_prime, rho = rho)[1])
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
    if true && emp_data.n > 2 * p
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

  (vars_result, status) = admm(emp_data, admm_data, solvers, dual_update, vars, compute_bmap = compute_bmap)
  B_admm = vec2mat(vars_result[end], emp_data)
  #= println(vecnorm(vars_result[end] - mat2vec(B))) =#
  return (B_admm, status)
end

function min_admm_oracle(pop_data, emp_data, admm_data, lambdas)
  errors = zeros(length(lambdas))
  Bs = []
  B_admm = copy(admm_data.B0)
  status = zeros(length(lambdas))
  for i = 1:length(lambdas)
    println("lambda = ", lambdas[i])
    admm_data.B0 = copy(B_admm)
    admm_data.quic_data.lambda = lambdas[i]
    result = min_admm(emp_data, admm_data)
    B_admm = copy(result[1])
    status[i] = result[2]
    push!(Bs, B_admm)
    errors[i] = vecnorm(B_admm - pop_data.B)
  end

  (err, ind) = findmin(errors)
  return (Bs[ind], err, lambdas[ind], errors, status)
end

function min_constr_lh_oracle(pop_data, emp_data, lh_data, lambdas)
  errors = zeros(length(lambdas))
  Bs = []
  B_lh = copy(lh_data.B0)
  for i = 1:length(lambdas)
    lh_data.lambda = lambdas[i]
    if lh_data.continuation
      lh_data.B0 = copy(B_lh)
    end
    if lh_data.use_constraint
      B_lh = copy(min_constraint_lh(emp_data, lh_data))
    else
      B_lh = copy(min_vanilla_lh(emp_data, lh_data))
    end
    push!(Bs, B_lh)
    errors[i] = vecnorm(B_lh - pop_data.B)
  end

  (err, ind) = findmin(errors)
  return (Bs[ind], err, lambdas[ind], errors)
end

function combined_oracle(pop_data, emp_data, admm_data, lh_data, lambdas)
  (B1, err1, lambda1, errors1, status1) = min_admm_oracle(pop_data, emp_data, admm_data, lambdas)
  lh_data.x_base = mat2vec(B1, emp_data)
  lh_data.upper_bound = vecnorm(pop_data.B - B1)^2
  lh_data.B0 = copy(B1)
  (B2, err2, lambda2, errors2) = min_constr_lh_oracle(pop_data, emp_data, lh_data, lambdas) 
  return (B1, B2, err1, err2, lambda1, lambda2, errors1, errors2, status1)
end

function min_admm_cv(pop_data, emp_data, admm_data, lambdas, k)
  lhs = zeros(length(lambdas))
  B0_old = copy(admm_data.B0)
  B_admms = [copy(admm_data.B0) for i = 1:k]
  emp_datas_train = []
  emp_datas_test = []
  p = pop_data.p

  lh_data = VanillaLHData(p, 0, zeros(p, p))

  for i = 1:k
    (emp_train, emp_test) = k_fold_split(pop_data, emp_data, k, i)
    push!(emp_datas_train, emp_train)
    push!(emp_datas_test, emp_test)
  end

  for i = 1:length(lambdas)
    println("lambda = ", lambdas[i])
    lh_val = 0
    for j = 1:k
      admm_data.B0 = copy(B_admms[j])
      admm_data.quic_data.lambda = lambdas[i]
      B_admms[j] = copy(min_admm(emp_datas_train[j], admm_data)[1])
      println(size(B_admms[j]))
      lh_val += lh(emp_datas_test[j], lh_data, mat2vec(B_admms[j], emp_data, reduced = true), [])
    end
    lhs[i] = lh_val
  end

  (lh_val, ind) = findmin(lhs)
  admm_data.B0 = B0_old
  admm_data.quic_data.lambda = lambdas[ind]
  B = min_admm(emp_data, admm_data)[1]
  return (B, lh_val, lambdas[ind], lhs)
end

function min_constr_lh_cv(pop_data, emp_data, lh_data, lambdas, k)
  lhs = zeros(length(lambdas))
  B_lhs = [copy(lh_data.B0) for i =1:k]
  B0_old = copy(lh_data.B0)

  emp_datas_train = []
  emp_datas_test = []
  p = pop_data.p

  for i = 1:k
    (emp_train, emp_test) = k_fold_split(pop_data, emp_data, k, i)
    push!(emp_datas_train, emp_train)
    push!(emp_datas_test, emp_test)
  end

  for i = 1:length(lambdas)
    println("lambda = ", lambdas[i])
    lh_val = 0
    lh_data.lambda = lambdas[i]
    for j = 1:k
      if lh_data.continuation
        lh_data.B0 = copy(B_lhs[j])
      end
      if lh_data.use_constraint
        B_lhs[j] = copy(min_constraint_lh(emp_datas_train[j], lh_data))
      else
        B_lh[j] = copy(min_vanilla_lh(emp_data_datas_train[j], lh_data))
      end
      lh_val += lh(emp_datas_test[j], lh_data, mat2vec(B_lhs[j], emp_data, reduced = true), [])
    end
    lhs[i] = lh_val
  end


  (lh_val, ind) = findmin(lhs)
  lh_data.B0 = B0_old
  lh_data.lambda = lambdas[ind]
  if lh_data.use_constraint
    B = copy(min_constraint_lh(emp_data, lh_data))
  else
    B = copy(min_vanilla_lh(emp_data_data, lh_data))
  end

  return (B, lh_val, lambdas[ind], lhs)
end

function combined_cv(pop_data, emp_data, admm_data, lh_data, lambdas, k, c)
  (B1, lh1, lambda1, lhs1) = min_admm_cv(pop_data, emp_data, admm_data, lambdas, k)
  lh_data.x_base = mat2vec(B1, emp_data)
  lh_data.upper_bound = c * min(1/pop_data.p, 1/emp_data.E)
  lh_data.B0 = copy(B1)
  (B2, lh2, lambda2, lhs2) = min_constr_lh_cv(pop_data, emp_data, lh_data, lambdas, k) 
  return (B1, B2, lh1, lh2, lambda1, lambda2, lhs1, lhs2)
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
