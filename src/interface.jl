# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    StatsLearnModel(model, invars, outvars)

Wrap a (possibly external) `model` with selectors of
input variables `invars` and output variables `outvars`.

## Examples

```julia
StatsLearnModel(DecisionTreeClassifier(), ["x1","x2"], "y")
StatsLearnModel(DecisionTreeClassifier(), 1:3, "target")
```
"""
struct StatsLearnModel{M,I<:ColumnSelector,O<:ColumnSelector}
  model::M
  invars::I
  outvars::O
end

StatsLearnModel(model, invars, outvars) = StatsLearnModel(model, selector(invars), selector(outvars))

"""
    fit(model, input, output)

Fit statistical learning `model` using features in `input` table
and targets in `output` table. Returns a fitted model with all
the necessary information for prediction with the `predict` function.
"""
function fit end

function Base.show(io::IO, model::StatsLearnModel{M}) where {M}
  println(io, "StatsLearnModel{$(nameof(M))}")
  println(io, "├─ features: $(model.invars)")
  print(io, "└─ targets: $(model.outvars)")
end

"""
    FittedStatsLearnModel(model, cache)

Wrap the statistical learning `model` with the `cache`
produced during the [`fit`](@ref) stage.
"""
struct FittedStatsLearnModel{M,C}
  model::M
  cache::C
end

"""
    predict(model::FittedStatsLearnModel, table)

Predict targets using the fitted statistical
learning `model` and a new `table` of features.
"""
function predict end

Base.show(io::IO, ::FittedStatsLearnModel{M}) where {M} = print(io, "FittedStatsLearnModel{$(nameof(M))}")
