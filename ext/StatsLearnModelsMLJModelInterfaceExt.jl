module StatsLearnModelsMLJModelInterfaceExt

using Tables
import StatsLearnModels as SLM
import MLJModelInterface as MI

isprobabilistic(model::MI.Model) = MI.prediction_type(model) == :probabilistic
isprobabilistic(model::MI.Probabilistic) = true

function SLM.fit(model::MI.Model, input, output)
  cols = Tables.columns(output)
  names = Tables.columnnames(cols)
  y = Tables.getcolumn(cols, first(names))
  data = MI.reformat(model, input, y)
  fitresult, _... = MI.fit(model, 0, data...)
  SLM.FittedModel(model, fitresult)
end

function SLM.predict(fmodel::SLM.FittedModel{<:MI.Model}, table)
  (; model, fitresult) = fmodel
  data = MI.reformat(model, table)
  if isprobabilistic(model)
    MI.predict_mode(model, fitresult, data...)
  else
    MI.predict(model, fitresult, data...)
  end
end

end
