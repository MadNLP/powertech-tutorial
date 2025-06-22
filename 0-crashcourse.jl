
# # Crash-course
# This first part is devoted to learning the basic of GPU programming.
# For those not familiar with the Julia language, we highly recommend
# reading [this introduction](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_julia/).

# ## GPU programming using CUDA.jl
# Julia has an excellent support for GPU programming with the package CUDA.jl.
# We recommend [this tutorial](https://cuda.juliagpu.org/stable/tutorials/introduction/)
# to understand the basic concept for programming on the GPU. Once installed,
# you can import CUDA as:

using CUDA

# By default, Julia allocates a new array on the host:
x_cpu = zeros(10)

# The allocation of a new vector on the GPU has to be explicited as
x_gpu = CUDA.zeros(10)

# By default, CUDA.jl allocates a vector of `float`. In scientific computing, it
# is often recommended to work with `double`, encoded by the type `Float64` in Julia:
x_gpu = CUDA.zeros(Float64, 10)

# The array can be manipulated using the broadcast operator (using a syntax similar as in matlab).
# Incrementing all the elements in `x_gpu` by 1 just amounts to
x_gpu .+= 1.0

# On the GPU, accessing the element of an array by its index (e.g. calling `x_gpu[1]`)
# is prohibited by default, and should by avoided
# at all cost. The whole point of using a GPU is to evaluate things in parallel, so it usually makes little
# sense to access an array element by element. If you have to implement non-trivial operations with complicated indexing, you have to resort to implementing a custom GPU kernels by yourself. In general,
#
# - We recommend using the broadcast operator `.` as much as you can, as it generates automatically the GPU kernels you need to implement the operation.
# - If you really have to, you can implement your own GPU kernel [using CUDA.jl](https://cuda.juliagpu.org/stable/tutorials/introduction/#Writing-your-first-GPU-kernel) or using an abstraction layer like [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl/).
#
# Now comes the question of evaluating complicated expressions on the GPU.

# ## Modeling with ExaModels.jl
# In optimization, it is recommended to use a modeler that acts as a domain specific language providing you all the syntax needed to implement your optimization problem. [Ampl](https://ampl.com), [JuMP.jl](https://jump.dev) and [Pyomo](https://www.pyomo.org/) are among the most popular modelers, but none of them support GPUs.
# [ExaModels.jl](https://exanauts.github.io/ExaModels.jl/dev/) is an attempt to fill this gap.
# As for CUDA.jl, we recommend [this introductory course](https://exanauts.github.io/ExaModels.jl/dev/guide/) to ExaModels.

# You can import ExaModels.jl simply as
using ExaModels

# and instantiate a new model with
core = ExaCore()

# Adding new variables to the model is very much similar to other modelers. E.g., adding 10 lower-bounded
# variables ``x ≥ 0`` amounts to
x = variable(core, 10; lvar=0.0)

# Once the variables defined, ExaModels relies on [a powerful SIMD abstraction](https://exanauts.github.io/ExaModels.jl/dev/simd/) to identify
# automatically the potential for parallelism in the expression tree.
# ExaModels implements the expression trees using iterator objects. Like programming on GPU,
# we should define all the expression in iterator format, and we should avoid accessing the variable `x`
# by its index outside a generator.
#
# As a demonstration, we show how to generate the constraint ``10 × sin(x_i) ≥ 0``.
# We start by building a Julia generator encoding the expression:
gen = (10.0 * sin(x[i]) + i for i in 1:10)

# We can pass the generator `gen` to ExaModels to build our inequality constraint:
cons = constraint(core, gen; lcon=0.0)

# Note that the generator does not evaluate the expression, but just provide a way to generate it.
# The evaluation part comes apart, by creating an `ExaModel` instance that takes as input
# the structure `core` that stores all the expressions used to generate the model:
nlp = ExaModel(core)

# The constructor `ExaModel` generates an `AbstractNLPModel`, which comes with a proper API
# to evaluate the model in a syntax appropriate for numerical optimization.
# The API can be found [in this documentation](https://jso.dev/NLPModels.jl/stable/api/#Reference-guide).
# As a consequence, evaluating the constraints implemented by the generator we defined before just
# translates to:
using NLPModels
x = ones(10)  # get an initial point
c = NLPModels.cons(nlp, x)  # return the results as a vector


# As ExaModels is just manipulating expression, it is very easy to offload the evaluation of the model on the GPU: it just requires to build the appropriate kernels to evaluate the expressions implemented in the generators, a task performed automatically by ExaModels.
# You can generate a new model on the GPU simply by specifying a new backend to ExaModels:
core = ExaCore(; backend=CUDABackend())

# Afterwards, the generation of the model remains the same:
x = variable(core, 10; lvar=0.0)
cons = constraint(core, 10.0 * sin(x[i]) + i for i in 1:10; lcon=0.0)

# And the evaluation of the model follows exactly the same syntax:

nlp = ExaModel(core)
x_gpu = CUDA.ones(Float64, 10)  # get an initial point
c_gpu = NLPModels.cons(nlp, x_gpu)  # return the results as a vector

# As we will see in the next tutorial, ExaModels is a powerful tool to evaluate the
# model's derivatives using automatic differentiation. This will prove to be particularly
# useful for solving the power flow equations.

