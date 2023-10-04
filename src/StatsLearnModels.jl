# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

module StatsLearnModels

using Tables
using ColumnSelectors: selector
using TableTransforms: StatelessFeatureTransform
import TableTransforms: applyfeat, isrevertible

include("interface.jl")
include("learn.jl")

export Learn

end
