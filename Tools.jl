
module Tools
export export_table, pp

using OdsIO

function export_table(table, fname = "peek.ods", pos = (1, 1), sheet = "Sheet1")
  if typeof(table) <: Vector
    table = reshape(table, length(table), 1)
  end
  ods_write(fname, Dict((sheet, pos...) => table))
end

function pp(mat)
  show(IOContext(STDOUT, limit=true), "text/plain", mat)
  println()
  println()
end

function matrix_grad(f, x0, epsilon = 1e-7)
	println(size(x0))
	(p1, p2) = size(x0)
	println(p1, " ", p2)
	ret = similar(x0)
	for i = 1:p1, j=1:p2
		E = zeros(p1, p2)
		E[i, j] = 1
		ret[i, j] = (f(x0 + epsilon * E) - f(x0))/epsilon
	end

	return ret
end

end
