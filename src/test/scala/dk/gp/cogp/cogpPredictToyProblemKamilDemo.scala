package dk.gp.cogp

import breeze.linalg._
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.numerics._
import breeze.plot.Figure
import breeze.plot.plot
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.calcLBLoglik
import dk.gp.cogp.svi.stochasticUpdateCogpModel
import dk.gp.cogp.testutils._
import dk.gp.cogp.testutils.createCogpToyModel
import java.io.File
import com.typesafe.scalalogging.slf4j.LazyLogging
import scala.math.Pi
import dk.gp.cogp.model.Task
import dk.bayes.math.gaussian.Gaussian

object cogpPredictToyProblemKamilDemo extends App with LazyLogging {

  val data = getData()
  val z = DenseVector.rangeD(0, 2 * Pi, 0.5).toDenseMatrix.t // inducing points for u and v inducing variables

  val now = System.currentTimeMillis()
  val initialToyModel = createCogpToyKamilModel(data, z)
  val trainedToyModel = cogpTrain(data, initialToyModel, iterNum = 500)
  logger.info("Total training time [ms]: " + (System.currentTimeMillis() - now))

  val xTest = DenseVector.rangeD(0, 2 * Pi, 0.1).toDenseMatrix.t + 0.001

  val predictedY0: DenseVector[Gaussian] = cogpPredict(xTest, i = 0, trainedToyModel)
  val predictedY1: DenseVector[Gaussian] = cogpPredict(xTest, i = 1, trainedToyModel)

  plotPredictions()

  private def getData(): Array[Task] = {
    val noise = Gaussian(0, 1)

    val x0 = DenseVector((0d to 2 * Pi by 0.1).filter(x => x < 1 || x > 2 * Pi - 1).toArray)
    val y0 = x0.map { x =>
      if (x < 1) sin(x) + 1e-7 + sqrt(1e-4) * noise.draw()
      else if (x > 2 * Pi - 1) -sin(x) + 1e-7 + sqrt(1e-4) * noise.draw()
      else throw new IllegalArgumentException("error")
    }

    val x1 = DenseVector.rangeD(0, 2 * Pi, 0.1)
    val y1 = x1.map { x => sin(x) + 1e-7 + sqrt(1e-4) * noise.draw() }

    val data = Array(Task(x0, y0), Task(x1, y1))

    data
  }

  private def plotPredictions() = {

    val predictedOutput1 = predictedY0.map(_.m)
    val predictedOutput2 = predictedY1.map(_.m)

    val figure = Figure()
    figure.subplot(0).legend = true

    figure.subplot(0) += plot(data(0).x.toDenseVector, data(0).y, '.', name = "actual output 1")
    figure.subplot(0) += plot(xTest.toDenseVector, predictedOutput1, name = "predicted output 1")

    figure.subplot(0) += plot(data(1).x.toDenseVector, data(1).y, '.', name = "actual output 2")
    figure.subplot(0) += plot(xTest.toDenseVector, predictedOutput2, name = "predicted output 2")

  }
}