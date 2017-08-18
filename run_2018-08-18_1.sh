#! /bin/bash

module load julia
parallel -j 2 'unbuffer julia5 CausalML/CausalMLTest.jl {} > CausalML/logs/log_{}.txt 2>&1' ::: {1..2}
