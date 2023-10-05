abstract type GLMModel end

struct LinearRegressor{K} <: GLMModel
  kwargs::K
end

LinearRegressor(; kwargs...) = LinearRegressor(values(kwargs))

struct GeneralizedLinearRegressor{D<:UnivariateDistribution,L<:Union{GLM.Link,Nothing},K} <: GLMModel
  dist::D
  link::L
  kwargs::K
end

GeneralizedLinearRegressor(dist::UnivariateDistribution, link=nothing; kwargs...) =
  GeneralizedLinearRegressor(dist, link, values(kwargs))

function fit(model::GLMModel, input, output)
  cols = Tables.columns(output)
  names = Tables.columnnames(cols)
  outcol = first(names)
  X = Tables.matrix(input)
  y = Tables.getcolumn(cols, outcol)
  fitted = _fit(model, X, y)
  FittedModel(model, (fitted, outcol))
end

function predict(fmodel::FittedModel{<:GLMModel}, table)
  model, outcol = fmodel.cache
  X = Tables.matrix(table)
  ŷ = GLM.predict(model, X)
  (; outcol => ŷ) |> Tables.materializer(table)
end

_fit(model::LinearRegressor, X, y) = GLM.lm(X, y; model.kwargs...)

function _fit(model::GeneralizedLinearRegressor, X, y)
  if isnothing(model.link) 
    GLM.glm(X, y, model.dist; model.kwargs...)
  else
    GLM.glm(X, y, model.dist, model.link; model.kwargs...)
  end
end
