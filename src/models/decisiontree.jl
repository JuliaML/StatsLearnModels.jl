# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

const DecisionTreeModel = Union{
  AdaBoostStumpClassifier,
  DecisionTreeClassifier,
  RandomForestClassifier,
  DecisionTreeRegressor,
  RandomForestRegressor
}

function fit(model::DecisionTreeModel, input, output)
  cols = Tables.columns(output)
  names = Tables.columnnames(cols)
  outnm = first(names)
  y = Tables.getcolumn(cols, outnm)
  X = Tables.matrix(input)
  DT.fit!(model, X, y)
  FittedModel(model, outnm)
end

function predict(fmodel::FittedModel{<:DecisionTreeModel}, table)
  outnm = fmodel.cache
  X = Tables.matrix(table)
  ŷ = DT.predict(fmodel.model, X)
  (; outnm => ŷ) |> Tables.materializer(table)
end
