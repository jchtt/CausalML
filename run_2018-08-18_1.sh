#! /bin/bash

module load julia
parallel -j 20 'unbuffer julia CausalML/CausalMLTest.jl {} &> CausalML/logs/log_{}.txt' ::: {1..32}
