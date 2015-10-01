package dk.gp.cogp

import breeze.linalg._
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.numerics._
import breeze.plot.Figure
import breeze.plot.plot
import breeze.stats.distributions.Gaussian
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.calcLBLoglik
import dk.gp.cogp.svi.stochasticUpdateCogpModel
import dk.gp.cogp.testutils._
import dk.gp.cogp.testutils.createCogpToyModel
import dk.gp.math.UnivariateGaussian
import java.io.File
import com.typesafe.scalalogging.slf4j.LazyLogging
import dk.gp.cogp.model.Task

//Long convergence, solution: change learning rate for g_u from 1e-2 to 0.5
//    val noise = Gaussian(0, 1)
//    val x = DenseVector.rangeD(-10, 10, 0.1)
//    val y0 = sin(x) + 1e-7 + sqrt(1e-4) * noise.draw()
//    val y1 = -sin(x) + 1e-7 + sqrt(1e-4) * noise.draw()
//    val y = DenseVector.horzcat(y0, y1)

object cogpPredictToyProblemDemo extends App with LazyLogging {

  val data: Array[Task] = loadToyModelDataIncomplete()
  val z = DenseVector.rangeD(-10, 10, 1).toDenseMatrix.t // inducing points for u and v inducing variables

  val now = System.currentTimeMillis()
  val initialToyModel = createCogpToyModel(data, z)
  val trainedToyModel = cogpTrain(data, initialToyModel, iterNum = 500)
  logger.info("Total training time [ms]: " + (System.currentTimeMillis() - now))

  val xTest = DenseVector.rangeD(-10, 10, 0.1)
  val predictedY: DenseMatrix[UnivariateGaussian] = cogpPredict(xTest + 0.001, trainedToyModel)

  plotPredictions()

  private def plotPredictions() = {

    val predictedOutput1 = predictedY(::, 0).map(_.m)
    val predictedOutput2 = predictedY(::, 1).map(_.m)

    val figure = Figure()
    figure.subplot(0).legend = true

    figure.subplot(0) += plot(data(0).x.toDenseVector, data(0).y, '.', name = "actual output 1")
    figure.subplot(0) += plot(xTest, predictedOutput1, name = "predicted output 1")

    figure.subplot(0) += plot(data(1).x.toDenseVector, data(1).y, '.', name = "actual output 2")
    figure.subplot(0) += plot(xTest, predictedOutput2, name = "predicted output 2")

  }
}