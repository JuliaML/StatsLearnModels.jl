using StatsLearnModels
using MLJ, MLJDecisionTreeInterface
using TableTransforms
using DataFrames
using Random
using Test

const SLM = StatsLearnModels

@testset "StatsLearnModels.jl" begin
  @testset "interface" begin
    Random.seed!(123)
    iris = DataFrame(load_iris())
    input = iris[:, Not(:target)]
    output = iris[:, [:target]]
    train, test = partition(1:nrow(input), 0.7, rng=123)
    Tree = @load(DecisionTreeClassifier, pkg=DecisionTree, verbosity=0)
    fmodel = SLM.fit(Tree(), input[train, :], output[train, :])
    pred = SLM.predict(fmodel, input[test, :])
    accuracy = count(pred.target .== output.target[test]) / length(test)
    @test accuracy > 0.9
  end

  @testset "Learn" begin
    Random.seed!(123)
    iris = DataFrame(load_iris())
    outcol = :target
    incols = setdiff(propertynames(iris), [outcol])
    train, test = partition(1:nrow(iris), 0.7, rng=123)
    Tree = @load(DecisionTreeClassifier, pkg=DecisionTree, verbosity=0)
    transform = Learn(iris[train, :], Tree(), incols => outcol)
    @test !isrevertible(transform)
    pred = transform(iris[test, :])
    accuracy = count(pred.target .== iris.target[test]) / length(test)
    @test accuracy > 0.9

    # throws
    # training data is not a table
    @test_throws ArgumentError Learn(nothing, Tree(), incols => outcol)
  end
end
