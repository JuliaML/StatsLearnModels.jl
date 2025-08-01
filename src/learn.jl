# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    Learn(train, model, features => targets)

Fits the statistical learning `model` to `train` table,
using the selectors of `features` and `targets`.

## Examples

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
  feats::Vector{Symbol}
end

Learn(train, model, (feats, targs)::Pair) = Learn(train, StatsLearnModel(model, feats, targs))

function Learn(train, lmodel::StatsLearnModel)
  if !Tables.istable(train)
    throw(ArgumentError("training data must be a table"))
  end

  cols = Tables.columns(train)
  names = Tables.columnnames(cols)
  feats = lmodel.feats(names)
  targs = lmodel.targs(names)

  input = (; (var => Tables.getcolumn(cols, var) for var in feats)...)
  output = (; (var => Tables.getcolumn(cols, var) for var in targs)...)

  fmodel = fit(lmodel.model, input, output)

  Learn(fmodel, feats)
end

isrevertible(::Type{<:Learn}) = false

function applyfeat(transform::Learn, feat, prep)
  model = transform.model
  vars = transform.feats

  cols = Tables.columns(feat)
  pairs = (var => Tables.getcolumn(cols, var) for var in vars)
  test = (; pairs...) |> Tables.materializer(feat)

  predict(model, test), nothing
end
