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

It is possible train a learning `model` (e.g. `RandomForestClassifier`) with
the `train` table to approximate a `:target` label and perform predictions
with the `test` table:

```julia
model = RandomForestClassifier()
learn = Learn(label(train, :target); model)
preds = learn(test)
```

The function `label` is used to tag columns of the table with target labels,
which can be categorical or continuous. All remaining columns are assumed to
be predictors.

The package exports native Julia models from various packages
in the ecosystem. It is also possible to use models from the
[MLJ.jl](https://github.com/JuliaAI/MLJ.jl) stack.

[build-img]: https://img.shields.io/github/actions/workflow/status/JuliaML/StatsLearnModels.jl/CI.yml?branch=main&style=flat-square
[build-url]: https://github.com/JuliaML/StatsLearnModels.jl/actions

[codecov-img]: https://img.shields.io/codecov/c/github/JuliaML/StatsLearnModels.jl?style=flat-square
[codecov-url]: https://codecov.io/gh/JuliaML/StatsLearnModels.jl
