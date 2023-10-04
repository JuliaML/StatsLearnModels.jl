# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    Learn(train, model, incols => outcols)

Fits the statistical learning `model` using the input columns, selected by `incols`,
and the output columns, selected by `outcols`, from the `train` table.

The column selection can be a single column identifier (index or name),
a collection of identifiers or a regular expression (regex).

# Examples

```julia
Learn(train, model, [1, 2, 3] => "d")
Learn(train, model, [:a, :b, :c] => :d)
Learn(train, model, ["a", "b", "c"] => 4)
Learn(train, model, [1, 2, 3] => [:d, :e])
Learn(train, model, r"[abc]" => ["d", "e"])
```
"""
struct Learn{M<:FittedModel} <: StatelessFeatureTransform
  model::M
  input::Vector{Symbol}
end

function Learn(train, model, (incols, outcols)::Pair)
  if !Tables.istable(train)
    throw(ArgumentError("training data must be a table"))
  end

  cols = Tables.columns(train)
  names = Tables.columnnames(cols)
  innms = selector(incols)(names)
  outnms = selector(outcols)(names)

  input = (; (nm => Tables.getcolumn(cols, nm) for nm in innms)...)
  output = (; (nm => Tables.getcolumn(cols, nm) for nm in outnms)...)

  fmodel = fit(model, input, output)
  Learn(fmodel, innms)
end

isrevertible(::Type{<:Learn}) = false

function applyfeat(transform::Learn, feat, prep)
  cols = Tables.columns(feat)
  pairs = (nm => Tables.getcolumn(cols, nm) for nm in transform.input)
  test = (; pairs...) |> Tables.materializer(feat)
  predict(transform.model, test), nothing
end
