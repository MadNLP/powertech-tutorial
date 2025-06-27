
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
        "Tutorial 2: Batched Power Flow" => "2-batched-powerflow.md",
        "Tutorial 3: Constrained Power Flow" => "3-constrained-powerflow.md",
        "Tutorial 4: Optimal Power Flow" => "4-optimal-powerflow.md",
        "Tutorial 5: ExaModelsPower.jl" => "5-exa-models-power.md",
    ]
)

deploydocs(repo = "github.com/MadNLP/powertech-tutorial.git"; push_preview = true)
