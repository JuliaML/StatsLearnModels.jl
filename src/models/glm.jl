# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

abstract type GLMModel end

"""
    LinearRegressor(; kwargs...)

Linear regression model.

The `kwargs` are forwarded to the `GLM.lm` function
from [GLM.jl](https://github.com/JuliaStats/GLM.jl).

See also [`GeneralizedLinearRegressor`](@ref).
"""
struct LinearRegressor{K} <: GLMModel
  kwargs::K
end

LinearRegressor(; kwargs...) = LinearRegressor(values(kwargs))

"""
    GeneralizedLinearRegressor(dist, link; kwargs...)

Generalized linear regression model with distribution `dist`
from Distributions.jl and `link` function.

The `kwargs` are forwarded to the `GLM.glm` function
from [GLM.jl](https://github.com/JuliaStats/GLM.jl).

See also [`LinearRegressor`](@ref).
"""
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
  outnm = first(names)
  X = Tables.matrix(input)
  y = Tables.getcolumn(cols, outnm)
  fitted = _fit(model, X, y)
  FittedStatsLearnModel(model, (fitted, outnm))
end

function predict(fmodel::FittedStatsLearnModel{<:GLMModel}, table)
  model, outnm = fmodel.cache
  X = Tables.matrix(table)
  ŷ = GLM.predict(model, X)
  (; outnm => ŷ) |> Tables.materializer(table)
end

_fit(model::LinearRegressor, X, y) = GLM.lm(X, y; model.kwargs...)

function _fit(model::GeneralizedLinearRegressor, X, y)
  if isnothing(model.link) 
    GLM.glm(X, y, model.dist; model.kwargs...)
  else
    GLM.glm(X, y, model.dist, model.link; model.kwargs...)
  end
end
