package dk.gp.cogp

import breeze.linalg.DenseMatrix
import breeze.plot.Figure
import breeze.plot.plot
import dk.gp.cogp.testutils.createCogpToyModel
import dk.gp.cogp.testutils.loadToyModelDataIncomplete
import dk.gp.math.UnivariateGaussian

object cogpPredictToyProblemDemo extends App {

  val data: DenseMatrix[Double] = loadToyModelDataIncomplete()

  val x = data(::, 0).toDenseMatrix.t
  val y = data(::, 1 to 2)

  val z = x(0 until x.rows by 10, ::) // inducing points for u and v inducing variables

  val now = System.currentTimeMillis()
  val initialToyModel = createCogpToyModel(x, y, z)
  val trainedToyModel = cogpTrain(x, y, initialToyModel, iterNum = 500)
  println(System.currentTimeMillis() - now)

  val predictedY: DenseMatrix[UnivariateGaussian] = cogpPredict(x, trainedToyModel)

  plotPredictions()

  private def plotPredictions() = {

    val predictedOutput1 = predictedY(::, 0).map(_.m)
    val predictedOutput2 = predictedY(::, 1).map(_.m)

    val figure = Figure()
    figure.subplot(0).legend = true

    figure.subplot(0) += plot(x(::, 0), y(::, 0), '.', name = "actual output 1")
    figure.subplot(0) += plot(x(::, 0), predictedOutput1, name = "predicted output 1")

    figure.subplot(0) += plot(x(::, 0), y(::, 1), '.', name = "actual output 2")
    figure.subplot(0) += plot(x(::, 0), predictedOutput2, name = "predicted output 2")

  }
}