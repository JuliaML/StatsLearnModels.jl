# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

module StatsLearnModels

using Distances
using PrettyTables
using StyledStrings
using DataScienceTraits
using StatsBase: mode, mean
using ColumnSelectors: ColumnSelector, selector
using TableTransforms: StatelessFeatureTransform

import Tables
import TableTransforms: applyfeat, isrevertible

using DecisionTree: AdaBoostStumpClassifier, DecisionTreeClassifier, RandomForestClassifier
using DecisionTree: DecisionTreeRegressor, RandomForestRegressor
using Distributions: UnivariateDistribution
using NearestNeighbors: MinkowskiMetric

import GLM
import DecisionTree as DT
import NearestNeighbors as NN

include("labeledtable.jl")
include("interface.jl")
include("models/nn.jl")
include("models/glm.jl")
include("models/tree.jl")
include("learn.jl")

export
  # labeled table
  LabeledTable,
  label,

  # NearestNeighbors.jl
  KNNClassifier,
  KNNRegressor,

  # GLM.jl
  LinearRegressor,
  GeneralizedLinearRegressor,

  # DecisionTree.jl
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  RandomForestClassifier,
  RandomForestRegressor,
  AdaBoostStumpClassifier,
  
  # transform
  Learn

end
