# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

abstract type NearestNeighborsModel end

struct KNNClassifier{M<:Metric} <: NearestNeighborsModel
  k::Int
  metric::M
  leafsize::Int
  reorder::Bool
end

KNNClassifier(k, metric=Euclidean(); leafsize=10, reorder=true) = KNNClassifier(k, metric, leafsize, reorder)

struct KNNRegressor{M<:Metric} <: NearestNeighborsModel
  k::Int
  metric::M
  leafsize::Int
  reorder::Bool
end

KNNRegressor(k, metric=Euclidean(); leafsize=10, reorder=true) = KNNRegressor(k, metric, leafsize, reorder)

function fit(model::NearestNeighborsModel, input, output)
  cols = Tables.columns(output)
  outnm = Tables.columnnames(cols) |> first
  outcol = Tables.getcolumn(cols, outnm)
  _checkoutput(model, outcol)
  (; metric, leafsize, reorder) = model
  data = Tables.matrix(input, transpose=true)
  tree = if metric isa MinkowskiMetric
    NN.KDTree(data, metric; leafsize, reorder)
  else
    NN.BallTree(data, metric; leafsize, reorder)
  end
  FittedStatsLearnModel(model, (tree, outnm, outcol))
end

function predict(fmodel::FittedStatsLearnModel{<:NearestNeighborsModel}, table)
  (; model, cache) = fmodel
  tree, outnm, outcol = cache
  data = Tables.matrix(table, transpose=true)
  indvec, _ = NN.knn(tree, data, model.k)
  aggfun = _aggfun(model)
  ŷ = [aggfun(outcol[inds]) for inds in indvec]
  (; outnm => ŷ) |> Tables.materializer(table)
end

function _checkoutput(::KNNClassifier, x)
  if !(elscitype(x) <: Categorical)
    throw(ArgumentError("output column must be categorical"))
  end
end

function _checkoutput(::KNNRegressor, x)
  if !(elscitype(x) <: Continuous)
    throw(ArgumentError("output column must be continuous"))
  end
end

_aggfun(::KNNClassifier) = mode
_aggfun(::KNNRegressor) = mean
