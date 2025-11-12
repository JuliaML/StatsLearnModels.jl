# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    Learn(table; [model])

Perform supervised learning with labeled `table` and
statistical learning `model`.

Uses `KNNClassifier(1)` or `KNNRegressor(1)` model by
default depending on the scientific type of the labels
stored in the table.

## Examples

```julia
Learn(label(table, "y"))
Learn(label(table, ["y1", "y2"]))
Learn(label(table, 3), model=KNNClassifier(5))
```

See also [`label`](@ref).
"""
struct Learn{T<:LabeledTable,M} <: StatelessFeatureTransform
  table::T
  model::M
end

Learn(table::LabeledTable; model=_defaultmodel(table)) = Learn(table, model)

function applyfeat(transform::Learn, feat, prep)
  # labeled table and model
  table = transform.table
  model = transform.model

  # variables in labeled table
  cols = Tables.columns(table)
  vars = Tables.columnnames(cols)

  # split targets and predictors
  targs = table.labels
  preds = setdiff(vars, targs)

  # learn function with statistical model
  input = (; (pred => Tables.getcolumn(cols, pred) for pred in preds)...)
  output = (; (targ => Tables.getcolumn(cols, targ) for targ in targs)...)
  fmodel = fit(model, input, output)

  # predict labels with new predictors
  fcols = Tables.columns(feat)
  fvars = Tables.columnnames(fcols)
  preds ⊆ fvars || throw(ArgumentError("predictors $preds not found in input table"))
  finput = (; (pred => Tables.getcolumn(fcols, pred) for pred in preds)...)
  foutput = predict(fmodel, finput) |> Tables.materializer(feat)

  foutput, nothing
end

function _defaultmodel(table::LabeledTable)
  cols = Tables.columns(table)
  vals = Tables.getcolumn(cols, only(table.labels))
  type = elscitype(vals)
  if type <: Categorical
    KNNClassifier(1)
  elseif type <: Continuous
    KNNRegressor(1)
  else
    throw(ErrorException("no default learning model for $type labels"))
  end
end
