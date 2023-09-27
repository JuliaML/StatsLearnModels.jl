module StatsLearnModels

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

end
