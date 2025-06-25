using Literate

dest_dir = "docs/src"
for script in ["utils.jl", "powerflow.jl"]
    cp(script, joinpath(dest_dir, script); force=true)
end
Literate.markdown("0-crashcourse.jl", dest_dir)
Literate.markdown("1-powerflow.jl", dest_dir)
Literate.markdown("2-block-powerflow.jl", dest_dir)
Literate.markdown("3-constrained-powerflow.jl", dest_dir)
Literate.markdown("4-optimal-powerflow.jl", dest_dir)

dest_dir = "notebooks"
for script in ["utils.jl", "powerflow.jl"]
    cp(script, joinpath(dest_dir, script); force=true)
end
Literate.notebook("0-crashcourse.jl", dest_dir)
Literate.notebook("1-powerflow.jl", dest_dir)
Literate.notebook("2-block-powerflow.jl", dest_dir)
Literate.notebook("3-constrained-powerflow.jl", dest_dir)
Literate.notebook("4-optimal-powerflow.jl", dest_dir)

include("docs/make.jl")
