#! /bin/bash

parallel -j 2 'unbuffer julia5 CausalML/CausalMLTest.jl {} > log_{}.txt 2>&1' ::: {1..2}
