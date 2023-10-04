const DTModel = Union{
  AdaBoostStumpClassifier,
  DecisionTreeClassifier,
  RandomForestClassifier,
  DecisionTreeRegressor,
  RandomForestRegressor
}

function fit(model::DTModel, input, output)
  cols = Tables.columns(output)
  names = Tables.columnnames(cols)
  outcol = first(names)
  y = Tables.getcolumn(cols, outcol)
  X = Tables.matrix(input)
  DT.fit!(model, X, y)
  FittedModel(model, outcol)
end

function predict(fmodel::FittedModel{<:DTModel}, table)
  outcol = fmodel.cache
  X = Tables.matrix(table)
  ŷ = DT.predict(fmodel.model, X)
  (; outcol => ŷ) |> Tables.materializer(table)
end
