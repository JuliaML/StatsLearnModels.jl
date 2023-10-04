using StatsLearnModels
using TableTransforms
using DataFrames
using Random
using Test

import MLJ, MLJDecisionTreeInterface

const SLM = StatsLearnModels

@testset "StatsLearnModels.jl" begin
  iris = DataFrame(MLJ.load_iris())
  input = iris[:, Not(:target)]
  output = iris[:, [:target]]
  train, test = MLJ.partition(1:nrow(input), 0.7, rng=123)

  @testset "interface" begin
    @testset "MLJ" begin
      Random.seed!(123)
      Tree = MLJ.@load(DecisionTreeClassifier, pkg=DecisionTree, verbosity=0)
      fmodel = SLM.fit(Tree(), input[train, :], output[train, :])
      pred = SLM.predict(fmodel, input[test, :])
      accuracy = count(pred.target .== output.target[test]) / length(test)
      @test accuracy > 0.9
    end

    @testset "DecisionTree" begin
      Random.seed!(123)
      model = DecisionTreeClassifier()
      fmodel = SLM.fit(model, input[train, :], output[train, :])
      pred = SLM.predict(fmodel, input[test, :])
      accuracy = count(pred.target .== output.target[test]) / length(test)
      @test accuracy > 0.9
    end
  end

  @testset "Learn" begin
    Random.seed!(123)
    outcol = :target
    incols = setdiff(propertynames(iris), [outcol])
    model = DecisionTreeClassifier()
    transform = Learn(iris[train, :], model, incols => outcol)
    @test !isrevertible(transform)
    pred = transform(iris[test, :])
    accuracy = count(pred.target .== iris.target[test]) / length(test)
    @test accuracy > 0.9

    # throws
    # training data is not a table
    @test_throws ArgumentError Learn(nothing, model, incols => outcol)
  end
end
