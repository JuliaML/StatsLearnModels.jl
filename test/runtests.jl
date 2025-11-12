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
  @testset "Basic" begin
    # show method
    x1 = rand(1:0.1:10, 100)
    x2 = rand(1:0.1:10, 100)
    y = 2x1 + x2
    input = DataFrame(; x1, x2)
    output = DataFrame(; y)
    train, test = 1:70, 71:100
    model = DecisionTreeClassifier()
    fmodel = SLM.fit(model, input[train, :], output[train, :])
    @test sprint(show, fmodel) == "FittedStatsLearnModel{DecisionTreeClassifier}"
  end

  @testset "Models" begin
    @testset "NearestNeighbors" begin
      Random.seed!(123)
      iris = DataFrame(MLJ.load_iris())
      input = iris[:, Not(:target)]
      output = iris[:, [:target]]
      train, test = MLJ.partition(1:nrow(input), 0.7, rng=123)
      model = KNNClassifier(5)
      fmodel = SLM.fit(model, input[train, :], output[train, :])
      pred = SLM.predict(fmodel, input[test, :])
      accuracy = count(pred.target .== output.target[test]) / length(test)
      @test accuracy > 0.9

      Random.seed!(123)
      x1 = rand(1:0.1:10, 100)
      x2 = rand(1:0.1:10, 100)
      y = 2x1 + x2
      input = DataFrame(; x1, x2)
      output = DataFrame(; y)
      model = KNNRegressor(5)
      fmodel = SLM.fit(model, input, output)
      pred = SLM.predict(fmodel, input)
      @test count(isapprox.(pred.y, y, atol=0.8)) > 80

      @test_throws ArgumentError SLM.fit(KNNClassifier(5), input, output)
      @test_throws ArgumentError SLM.fit(KNNRegressor(5), input, rand('a':'z', 100))
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

    @testset "DecisionTree" begin
      Random.seed!(123)
      iris = DataFrame(MLJ.load_iris())
      input = iris[:, Not(:target)]
      output = iris[:, [:target]]
      train, test = MLJ.partition(1:nrow(input), 0.7, rng=123)
      model = DecisionTreeClassifier()
      fmodel = SLM.fit(model, input[train, :], output[train, :])
      pred = SLM.predict(fmodel, input[test, :])
      accuracy = count(pred.target .== output.target[test]) / length(test)
      @test accuracy > 0.9
    end
  end

  @testset "Learn" begin
    Random.seed!(123)
    iris = DataFrame(MLJ.load_iris())
    input = iris[:, Not(:target)]
    output = iris[:, [:target]]
    train, test = MLJ.partition(1:nrow(input), 0.7, rng=123)
    model = DecisionTreeClassifier()
    transform = Learn(label(iris[train, :], :target); model)
    @test !isrevertible(transform)
    pred = transform(iris[test, :])
    accuracy = count(pred.target .== iris.target[test]) / length(test)
    @test accuracy > 0.9
  end

  @testset "MLJ" begin
    Random.seed!(123)
    iris = DataFrame(MLJ.load_iris())
    input = iris[:, Not(:target)]
    output = iris[:, [:target]]
    train, test = MLJ.partition(1:nrow(input), 0.7, rng=123)
    Tree = MLJ.@load(DecisionTreeClassifier, pkg = DecisionTree, verbosity = 0)
    fmodel = SLM.fit(Tree(), input[train, :], output[train, :])
    pred = SLM.predict(fmodel, input[test, :])
    accuracy = count(pred.target .== output.target[test]) / length(test)
    @test accuracy > 0.9
  end
end
