import StatsLearnModels as SLM
using MLJ, MLJDecisionTreeInterface
using DataFrames
using Random
using Test

@testset "StatsLearnModels.jl" begin
  Random.seed!(123)
  iris = DataFrame(load_iris())
  input = iris[:, Not(:target)]
  output = iris[:, [:target]]
  Tree = @load(DecisionTreeClassifier, pkg=DecisionTree, verbosity=0)
  train, test = partition(1:nrow(input), 0.7, rng=123)
  fmodel = SLM.fit(Tree(), input[train, :], output[train, :])
  pred = SLM.predict(fmodel, input[test, :])
  accuracy = count(pred.target .== output.target[test]) / length(test)
  @test accuracy > 0.9
end
