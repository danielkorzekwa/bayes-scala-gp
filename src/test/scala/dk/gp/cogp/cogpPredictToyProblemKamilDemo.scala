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
import scala.math.Pi

object cogpPredictToyProblemKamilDemo extends App with LazyLogging {

  val noise = Gaussian(0, 1)
  val x = DenseVector.rangeD(0, 2 * Pi, 0.1)
  val y0 = x.map { x =>
    if (x < 1) sin(x) + 1e-7 + sqrt(1e-4) * noise.draw()
    else if (x > 2 * Pi - 1) -sin(x) + 1e-7 + sqrt(1e-4) * noise.draw()
    else Double.NaN
  }
  val y1 = x.map { x => sin(x) + 1e-7 + sqrt(1e-4) * noise.draw() }

  val y = DenseVector.horzcat(y0, y1)
  val z = x(0 until x.size by 5) // inducing points for u and v inducing variables

  val now = System.currentTimeMillis()
  val initialToyModel = createCogpToyKamilModel(x, y, z)
  val trainedToyModel = cogpTrain(x, y, initialToyModel, iterNum = 500)
  logger.info("Total training time [ms]: " + (System.currentTimeMillis() - now))

  val predictedY: DenseMatrix[UnivariateGaussian] = cogpPredict(x + 0.001, trainedToyModel)

  plotPredictions()

  private def plotPredictions() = {

    val predictedOutput1 = predictedY(::, 0).map(_.m)
    val predictedOutput2 = predictedY(::, 1).map(_.m)

    val figure = Figure()
    figure.subplot(0).legend = true

    figure.subplot(0) += plot(x, y(::, 0), '.', name = "actual output 1")
    figure.subplot(0) += plot(x, predictedOutput1, name = "predicted output 1")

    figure.subplot(0) += plot(x, y(::, 1), '.', name = "actual output 2")
    figure.subplot(0) += plot(x, predictedOutput2, name = "predicted output 2")

  }
}