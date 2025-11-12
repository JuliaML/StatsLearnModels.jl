# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    fit(model, input, output)

Fit statistical learning `model` using predictors
in `input` table and targets in `output` table.
Returns a fitted model with all the necessary
information for prediction with [`predict`](@ref).
"""
function fit end

"""
    FittedStatsLearnModel(model, cache)

Wrap the statistical learning `model` with the
`cache` produced during the [`fit`](@ref) stage.
"""
struct FittedStatsLearnModel{M,C}
  model::M
  cache::C
end

"""
    predict(model::FittedStatsLearnModel, table)

Predict targets using the fitted statistical
learning `model` and a new `table` containing
the same predictors used during the [`fit`](@ref)
stage.
"""
function predict end

Base.show(io::IO, ::FittedStatsLearnModel{M}) where {M} = print(io, "FittedStatsLearnModel{$(nameof(M))}")
