# Collaborative multi-output Gaussian Process

Based on Nguyen et al. Collaborative Multi-output Gaussian Processes, 2014

## Toy example

**Observed data:**

*y1 = sin(x), y2 = -sin(x)*

**Prediction model:**

![cogp_toy_prediction_model](https://raw.github.com/danielkorzekwa/bayes-scala-gp/master/doc/cogp/cogp_toy_prediction_model.png)

**Training the model and predicting *y1(x)* and *y2(x)*,**
[source code](https://github.com/danielkorzekwa/bayes-scala-gp/blob/master/src/test/scala/dk/gp/cogp/cogpPredictToyProblemDemo.scala):

```scala
  val data: DenseMatrix[Double] = loadToyModelDataIncomplete()
  
  val x = data(::, 0).toDenseMatrix.t
  val y = data(::, 1 to 2)
  val z = x(0 until x.rows by 10, ::) // inducing points for u and v inducing variables
  
  val initialToyModel = createCogpToyModel(x, y, z)
  val trainedToyModel = cogpTrain(x, y, initialToyModel, iterNum = 500)

  val predictedY:DenseMatrix[UnivariateGaussian] = cogpPredict(x, trainedToyModel)
```

**Prediction plots**

![cogp_toy_problem_prediction_plot](https://raw.github.com/danielkorzekwa/bayes-scala-gp/master/doc/cogp/cogp_toy_problem_prediction_plot.png)
