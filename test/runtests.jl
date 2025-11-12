using StatsLearnModels
using TableTransforms
using DataFrames
using Random
using Test

using GLM: ProbitLink
using Distributions: Binomial

const SLM = StatsLearnModels

@testset "StatsLearnModels.jl" begin
  @testset "LabeledTable" begin
    # labels as symbols
    t = (x1=rand(3), x2=rand(3), y=rand(Int, 3))
    l = label(t, :y)
    @test parent(l) == t
    @test predictors(l) == [:x1, :x2]
    @test targets(l) == [:y]

    # labels as strings
    t = (x1=rand(3), x2=rand(3), y=rand(Int, 3))
    l = label(t, "y")
    @test parent(l) == t
    @test predictors(l) == [:x1, :x2]
    @test targets(l) == [:y]

    # multiple labels
    t = (x1=rand(3), x2=rand(3), y1=rand(Int, 3), y2=rand(Int, 3))
    l = label(t, ["y1", "y2"])
    @test parent(l) == t
    @test predictors(l) == [:x1, :x2]
    @test targets(l) == [:y1, :y2]

    # labels as regex
    t = (x1=rand(3), x2=rand(3), y1=rand(Int, 3), y2=rand(Int, 3))
    l = label(t, r"y")
    @test parent(l) == t
    @test predictors(l) == [:x1, :x2]
    @test targets(l) == [:y1, :y2]
  end

  @testset "Models" begin
    @testset "NearestNeighbors" begin
      Random.seed!(123)
      input = (x1=rand(100), x2=rand(100))
      output = (y=rand(1:3, 100),)
      model = KNNClassifier(5)
      fmodel = SLM.fit(model, input, output)
      foutput = SLM.predict(fmodel, input)
      accuracy = count(foutput.y .== output.y) / length(output.y)
      @test accuracy > 0

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
      foutput = SLM.predict(fmodel, input)
      @test all(isapprox.(foutput.y, output.y, atol=0.5))
      x = [1, 2, 2]
      y = [1, 0, 1]
      input = DataFrame(; ones=ones(length(x)), x)
      output = DataFrame(; y)
      model = GeneralizedLinearRegressor(Binomial(), ProbitLink())
      fmodel = SLM.fit(model, input, output)
      foutput = SLM.predict(fmodel, input)
      @test all(isapprox.(foutput.y, output.y, atol=0.5))
    end

    @testset "DecisionTree" begin
      Random.seed!(123)
      input = (x1=rand(100), x2=rand(100))
      output = (y=rand(1:3, 100),)
      model = DecisionTreeClassifier()
      fmodel = SLM.fit(model, input, output)
      foutput = SLM.predict(fmodel, input)
      accuracy = count(foutput.y .== output.y) / length(output.y)
      @test accuracy > 0
    end

    # show method
    x1 = rand(1:0.1:10, 100)
    x2 = rand(1:0.1:10, 100)
    y = 2x1 + x2
    input = DataFrame(; x1, x2)
    output = DataFrame(; y)
    model = DecisionTreeClassifier()
    fmodel = SLM.fit(model, input, output)
    @test sprint(show, fmodel) == "FittedStatsLearnModel{DecisionTreeClassifier}"
  end

  @testset "Learn" begin
    Random.seed!(123)
    train = (x1=rand(100), x2=rand(100), y=rand(1:3, 100))
    model = DecisionTreeClassifier()
    learn = Learn(label(train, :y); model)
    @test !isrevertible(learn)
    preds = learn(train)
    accuracy = count(preds.y .== train.y) / length(train.y)
    @test accuracy ≈ 1

    # default classification model
    Random.seed!(123)
    train = (x1=rand(100), x2=rand(100), y=rand(1:3, 100))
    learn = Learn(label(train, :y))
    @test !isrevertible(learn)
    preds = learn(train)
    accuracy = count(preds.y .== train.y) / length(train.y)
    @test accuracy ≈ 1

    # default regression model
    Random.seed!(123)
    train = (x1=rand(100), x2=rand(100), y=rand(100))
    learn = Learn(label(train, :y))
    @test !isrevertible(learn)
    preds = learn(train)
    error = sum(abs2, preds.y .- train.y) / length(train.y)
    @test error ≈ 0
  end
end
