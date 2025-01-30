# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

module StatsLearnModelsMLJModelInterfaceExt

using Tables
import StatsLearnModels as SLM
import MLJModelInterface as MI

isprobabilistic(model::MI.Model) = MI.prediction_type(model) == :probabilistic
isprobabilistic(model::MI.Probabilistic) = true

function SLM.fit(model::MI.Model, input, output)
  cols = Tables.columns(output)
  names = Tables.columnnames(cols)
  target = first(names)
  y = Tables.getcolumn(cols, target)
  data = MI.reformat(model, input, y)
  fitresult, _... = MI.fit(model, 0, data...)
  SLM.FittedStatsLearnModel(model, (fitresult, target))
end

function SLM.predict(fmodel::SLM.FittedStatsLearnModel{<:MI.Model}, table)
  (; model, cache) = fmodel
  fitresult, target = cache
  data = MI.reformat(model, table)
  ŷ = if isprobabilistic(model)
    MI.predict_mode(model, fitresult, data...)
  else
    MI.predict(model, fitresult, data...)
  end
  (; target => ŷ) |> Tables.materializer(table)
end

end
