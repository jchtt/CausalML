#! /bin/bash

parallel -j 16 'unbuffer julia CausalML/CausalMLTest.jl {} > log_{}.txt'
