module StatsLearnModels

using Tables
using ColumnSelectors: selector
using TableTransforms: StatelessFeatureTransform
import TableTransforms: applyfeat, isrevertible

export Learn

include("interface.jl")
include("learn.jl")

end
