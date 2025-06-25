
using Documenter

makedocs(
    sitename = "Powertech tutorial",
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX()
    ),
    checkdocs = :exports,
    clean=true,
    pages = [
        "Home" => "index.md",
        "Getting Started" => "0-crashcourse.md",
        "Tutorial 1: Power Flow" => "1-powerflow.md",
        "Tutorial 2: Block Power Flow" => "2-block-powerflow.md",
        "Tutorial 3: Constrained Power Flow" => "3-constrained-powerflow.md",
        "Tutorial 4: Optimal Power Flow" => "4-optimal-powerflow.md",
    ]
)

