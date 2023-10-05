# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

module StatsLearnModels

using Tables
using ColumnSelectors: selector
using TableTransforms: StatelessFeatureTransform
import TableTransforms: applyfeat, isrevertible

import GLM
import DecisionTree as DT
using DecisionTree: AdaBoostStumpClassifier, DecisionTreeClassifier, RandomForestClassifier
using DecisionTree: DecisionTreeRegressor, RandomForestRegressor
using Distributions: UnivariateDistribution

include("interface.jl")
include("models/decisiontree.jl")
include("models/glm.jl")
include("learn.jl")

export
  # models
  # DecisionTree
  AdaBoostStumpClassifier,
  DecisionTreeClassifier,
  RandomForestClassifier,
  DecisionTreeRegressor,
  RandomForestRegressor,
  # GLM
  LinearRegressor,
  GeneralizedLinearRegressor,

  # transform
  Learn

end
