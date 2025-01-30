# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    Learn(train, model, invars => outvars)

Fits the statistical learning `model` to `train` table,
using the selectors of input variables `invars` and
output variables `outvars`.

# Examples

```julia
Learn(train, model, [1, 2, 3] => "d")
Learn(train, model, [:a, :b, :c] => :d)
Learn(train, model, ["a", "b", "c"] => 4)
Learn(train, model, [1, 2, 3] => [:d, :e])
Learn(train, model, r"[abc]" => ["d", "e"])
```
"""
struct Learn{M<:FittedStatsLearnModel} <: StatelessFeatureTransform
  model::M
  invars::Vector{Symbol}
end

Learn(train, model, (invars, outvars)::Pair) = Learn(train, StatsLearnModel(model, invars, outvars))

function Learn(train, lmodel::StatsLearnModel)
  if !Tables.istable(train)
    throw(ArgumentError("training data must be a table"))
  end

  cols = Tables.columns(train)
  names = Tables.columnnames(cols)
  invars = lmodel.invars(names)
  outvars = lmodel.outvars(names)

  input = (; (var => Tables.getcolumn(cols, var) for var in invars)...)
  output = (; (var => Tables.getcolumn(cols, var) for var in outvars)...)

  fmodel = fit(lmodel.model, input, output)

  Learn(fmodel, invars)
end

isrevertible(::Type{<:Learn}) = false

function applyfeat(transform::Learn, feat, prep)
  model = transform.model
  vars = transform.invars

  cols = Tables.columns(feat)
  pairs = (var => Tables.getcolumn(cols, var) for var in vars)
  test = (; pairs...) |> Tables.materializer(feat)

  predict(model, test), nothing
end
