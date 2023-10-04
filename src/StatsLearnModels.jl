# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

module StatsLearnModels

using Tables
using ColumnSelectors: selector
using TableTransforms: StatelessFeatureTransform
import TableTransforms: applyfeat, isrevertible

import DecisionTree as DT
using DecisionTree: AdaBoostStumpClassifier, DecisionTreeClassifier, RandomForestClassifier
using DecisionTree: DecisionTreeRegressor, RandomForestRegressor

include("interface.jl")
include("models/decisiontree.jl")
include("learn.jl")

export Learn,
  AdaBoostStumpClassifier,
  DecisionTreeClassifier,
  RandomForestClassifier,
  DecisionTreeRegressor,
  RandomForestRegressor

end
