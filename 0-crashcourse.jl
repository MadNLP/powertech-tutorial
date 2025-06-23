
# # Getting started
# This first part is devoted to learning the basic of GPU programming.
# For those not familiar with the Julia language, we highly recommend
# reading [this introduction to Julia](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_julia/).

# ## GPU programming using CUDA.jl
# Julia has an excellent support for GPU programming with the organization [JuliaGPU](https://juliagpu.org/).
#
# In this tutorial, we will focus on the library CUDA provided by NVIDIA.
# We recommend reading [this tutorial](https://cuda.juliagpu.org/stable/tutorials/introduction/)
# to understand the basic concept for programming on the GPU with CUDA. Once installed, you can import CUDA as:

using CUDA

# By default, you can allocate a new vector in Julia using
x_cpu = zeros(10)

# The allocation of a new vector on the GPU has to be explicited as
x_gpu = CUDA.zeros(10)

# By default, CUDA.jl allocates a vector of `float`. In scientific computing, it
# is often recommended to work with `double`, encoded by the type `Float64` in Julia.
# This has to be explicited as
x_gpu = CUDA.zeros(Float64, 10)

# The array can be manipulated using the broadcast operator (using a syntax similar as in matlab).
# Incrementing all the elements in `x_gpu` by 1 just amounts to
x_gpu .+= 1.0

# !!! info
#     On the GPU, accessing the element of an array by its index (e.g. by calling `x_gpu[1]`)
#     is prohibited by default, and should by avoided at all cost. The whole point of using a GPU is to evaluate operations in parallel, so it usually makes little
#     sense to access an array element by element. If you have to implement non-trivial operations with complicated indexing operations,
#     it is recommended to implement your custom GPU kernels.
#
# In general,
#
# - We recommend using the broadcast operator `.` as much as you can, as it generates automatically the GPU kernels you need to implement the operation.
# - If you really have to, you can implement your own GPU kernel [using CUDA.jl](https://cuda.juliagpu.org/stable/tutorials/introduction/#Writing-your-first-GPU-kernel) or using an abstraction layer like [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl/).
#
# Now comes the question of evaluating complicated expressions on the GPU.

# ## Modeling with ExaModels.jl
# In optimization, it is recommended to use a modeler to implement your model. The modeler acts as a domain specific language providing you the syntax needed to implement your optimization problem, and
# it converts the resulting problem in a form suitable for an optimization solver.
# [Ampl](https://ampl.com), [JuMP.jl](https://jump.dev) and [Pyomo](https://www.pyomo.org/) are among the most popular modelers, but none of them support GPUs.
# [ExaModels.jl](https://exanauts.github.io/ExaModels.jl/dev/) is an attempt to fill this gap.
# ExaModels is designed to run on a variety of GPU architectures (NVIDIA, AMD, INTEL) as well as multi-threaded CPUs.
# As for CUDA.jl, we recommend [this introductory course](https://exanauts.github.io/ExaModels.jl/dev/guide/) to ExaModels.

# You can import ExaModels.jl simply as
using ExaModels

# and instantiate a new model with
core = ExaCore()

# Note that ExaModels supports multi-precision by default.
# Adding new variables to the model is very much similar to other modelers. E.g., adding 10 lower-bounded
# variables ``x ≥ 0`` amounts to
x = variable(core, 10; lvar=0.0)

# The keyword `lvar` is used to pass the lower-bounds on the variable `x`. Similarly, we can pass the upper-bounds using the keyword `uvar`,
# and a starting-point using the keyword `start`.

# Once the variables are defined, ExaModels relies on [a powerful SIMD abstraction](https://exanauts.github.io/ExaModels.jl/dev/simd/) to identify
# automatically the potential for parallelism in the expression tree.
# ExaModels implements the expression trees using iterator objects. As a consequence,
# we should define all the expressions in iterator format, and avoid accessing the variable `x` by its index outside a generator.
#
# As a demonstration, we show how to generate the constraint ``10 × sin(x_i) ≥ 0``.
# We start by building a Julia generator encoding the expression:
gen = (10.0 * sin(x[i]) + i for i in 1:10)

# We can pass the generator `gen` to ExaModels to build our inequality constraints:
cons = constraint(core, gen; lcon=0.0)

# We can add more iterators to the constraint `cons` by using the function `constraint!`.
#
# !!! info
#     All the data arrays should have concrete, bit-type elements.

# Note that the generator just provides a way to generate expression.
# The evaluation part comes apart, by creating an `ExaModel` instance that takes as input
# the structure `core` that stores all the generators required to build the model:
nlp = ExaModel(core)

# The constructor `ExaModel` generates an `AbstractNLPModel`, which comes with a proper API
# to evaluate the model in a syntax appropriate for numerical computing.
# The API can be found [in this documentation](https://jso.dev/NLPModels.jl/stable/api/#Reference-guide).
# As a consequence, evaluating the constraints implemented by the generator we defined before just translates to:
using NLPModels
x = ones(10)                # get an initial point
c = NLPModels.cons(nlp, x)  # return the results as a vector


# As ExaModels is just manipulating expressions, it is very easy to offload the evaluation of the model on the GPU.
# ExaModels builds automatically the appropriate kernels to evaluate the expressions implemented in the generators.
# You can generate a new model on the GPU simply by specifying a new backend to ExaModels:
core = ExaCore(; backend=CUDABackend())

# The generation of the model on the GPU follows the same syntax:
x = variable(core, 10; lvar=0.0)
cons = constraint(core, 10.0 * sin(x[i]) + i for i in 1:10; lcon=0.0)

# as well as the model's evaluation:

nlp = ExaModel(core)
x_gpu = CUDA.ones(Float64, 10)  # get an initial point
c_gpu = NLPModels.cons(nlp, x_gpu)  # return the results as a vector

# As we will see in the next tutorial, ExaModels is a powerful tool to evaluate the
# model's derivatives using automatic differentiation. This will prove to be particularly
# useful for solving the power flow equations.

