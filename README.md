# StatsLearnModels.jl

[![][build-img]][build-url] [![][codecov-img]][codecov-url]

Statistical learning models for [Tables.jl](https://github.com/JuliaData/Tables.jl) tables.

## Installation

Get the latest stable release with Julia's package manager:

```
] add StatsLearnModels
```

## Usage

This package provides a `Learn` transform that implements the
[TableTransforms.jl](https://github.com/JuliaML/TableTransforms.jl)
interface.

Given two [Tables.jl](https://github.com/JuliaData/Tables.jl)
tables with training and test data:

```julia
train = (feature1=rand(100), feature2=rand(100), target=rand(1:2, 100))
test = (feature1=rand(20), feature2=rand(20))
```

One can train a learning `model` (e.g. `RandomForestClassifier`) with
the `train` table:

```julia
model = RandomForestClassifier()

learn = Learn(train, model, ["feature1","feature2"] => "target")
```

and apply the trained `model` to the `test` table:

```julia
pred = learn(test)
```

The package exports native Julia models from various packages
in the ecosystem. It is also possible to use models from the
[MLJ.jl](https://github.com/JuliaAI/MLJ.jl) stack.

The combination of TableTransforms.jl with StatsLearnModels.jl
can be thought of as a powerful alternative to MLJ.jl.

[build-img]: https://img.shields.io/github/actions/workflow/status/JuliaML/StatsLearnModels.jl/CI.yml?branch=master&style=flat-square
[build-url]: https://github.com/JuliaML/StatsLearnModels.jl/actions

[codecov-img]: https://img.shields.io/codecov/c/github/JuliaML/StatsLearnModels.jl?style=flat-square
[codecov-url]: https://codecov.io/gh/JuliaML/StatsLearnModels.jl
