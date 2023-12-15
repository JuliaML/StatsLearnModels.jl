# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    StatsLearnModels.fit(model, input, output) -> FittedModel

Fit statistical learning `model` using features in `input` table
and targets in `output` table. Returns a fitted model with all
the necessary information for prediction with the `predict` function.
"""
function fit end

"""
    StatsLearnModels.predict(model::FittedModel, table)

Predict the target values using the fitted statistical learning `model`
and a new `table` of features.  
"""
function predict end

"""
    StatsLearnModels.FittedModel(model, cache)

Wrapper type used to save learning model and auxiliary
variables needed for prediction.
"""
struct FittedModel{M,C}
  model::M
  cache::C
end

Base.show(io::IO, ::FittedModel{M}) where {M} = print(io, "FittedModel{$(nameof(M))}")

"""
    StatsLearnModels.StatsLearnModel(model, incols, outcols)

Wrapper type for learning models used for dispatch purposes.
"""
struct StatsLearnModel{M,I<:ColumnSelector,O<:ColumnSelector}
  model::M
  input::I
  output::O
end

StatsLearnModel(model, incols, outcols) = StatsLearnModel(model, selector(incols), selector(outcols))

function Base.show(io::IO, model::StatsLearnModel{M}) where {M}
  println(io, "StatsLearnModel{$(nameof(M))}")
  println(io, "├─ input: $(model.input)")
  print(io, "└─ output: $(model.output)")
end

"""
    StatsLearnModels.model(lmodel::StatsLearnModel)
  
Returns the model of the `lmodel`.
"""
model(lmodel::StatsLearnModel) = lmodel.model

"""
    StatsLearnModels.input(lmodel::StatsLearnModel)
  
Returns the input column selection of the `lmodel`.
"""
input(lmodel::StatsLearnModel) = lmodel.input

"""
    StatsLearnModels.output(lmodel::StatsLearnModel)
  
Returns the output column selection of the `lmodel`.
"""
output(lmodel::StatsLearnModel) = lmodel.output
