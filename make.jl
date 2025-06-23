
using Literate

dest_dir = "docs/src"
Literate.markdown("0-crashcourse.jl", dest_dir)
Literate.markdown("1-powerflow.jl", dest_dir)
Literate.markdown("2-block-powerflow.jl", dest_dir)
Literate.markdown("3-constrained-powerflow.jl", dest_dir)
Literate.markdown("4-optimal-powerflow.jl", dest_dir)

# dest_dir = "notebooks"
# Literate.notebook("0-crashcourse.jl", dest_dir)
# Literate.notebook("1-powerflow.jl", dest_dir)
# Literate.notebook("2-block-powerflow.jl", dest_dir)
# Literate.notebook("3-constrained-powerflow.jl", dest_dir)
# Literate.notebook("4-optimal-powerflow.jl", dest_dir)
