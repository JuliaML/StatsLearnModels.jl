using StatsLearnModels
using TableTransforms
using DataFrames
using Random
using Test

using GLM: ProbitLink
using Distributions: Binomial

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
      Tree = MLJ.@load(DecisionTreeClassifier, pkg = DecisionTree, verbosity = 0)
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

    @testset "GLM" begin
      x = [1, 2, 3]
      y = [2, 4, 7]
      input = DataFrame(; ones=ones(length(x)), x)
      output = DataFrame(; y)
      model = LinearRegressor()
      fmodel = SLM.fit(model, input, output)
      pred = SLM.predict(fmodel, input)
      @test all(isapprox.(pred.y, output.y, atol=0.5))
      x = [1, 2, 2]
      y = [1, 0, 1]
      input = DataFrame(; ones=ones(length(x)), x)
      output = DataFrame(; y)
      model = GeneralizedLinearRegressor(Binomial(), ProbitLink())
      fmodel = SLM.fit(model, input, output)
      pred = SLM.predict(fmodel, input)
      @test all(isapprox.(pred.y, output.y, atol=0.5))
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
