module StatsLearnModels

"""
    StatsLearnModels.fit(model, input, output) -> FittedModel

TODO
"""
function fit end

"""
    StatsLearnModels.predict(model::FittedModel, table)

TODO
"""
function predict end

"""
    StatsLearnModels.FittedModel(model, fitresult)

TODO
"""
struct FittedModel{M,F}
  model::M
  fitresult::F
end

end
