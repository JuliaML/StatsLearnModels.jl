# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

module StatsLearnModels

using Tables
using Distances
using DataScienceTraits
using StatsBase: mode, mean
using ColumnSelectors: ColumnSelector, selector
using TableTransforms: StatelessFeatureTransform

import DataScienceTraits as DST
import TableTransforms: applyfeat, isrevertible

using DecisionTree: AdaBoostStumpClassifier, DecisionTreeClassifier, RandomForestClassifier
using DecisionTree: DecisionTreeRegressor, RandomForestRegressor
using Distributions: UnivariateDistribution
using NearestNeighbors: MinkowskiMetric

import GLM
import DecisionTree as DT
import NearestNeighbors as NN

include("interface.jl")
include("models/glm.jl")
include("models/decisiontree.jl")
include("models/nearestneighbors.jl")
include("learn.jl")

export
  # DecisionTree.jl
  AdaBoostStumpClassifier,
  DecisionTreeClassifier,
  RandomForestClassifier,
  DecisionTreeRegressor,
  RandomForestRegressor,
  
  # GLM.jl
  LinearRegressor,
  GeneralizedLinearRegressor,

  # NearestNeighbors.jl
  KNNClassifier,
  KNNRegressor,

  # transform
  Learn

end
