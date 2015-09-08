package dk.gp.cogp

import org.junit.Test
import breeze.linalg._
import java.io.File
import dk.gp.cogp.testutils.createCogpModel
import breeze.plot._

object cogpPredictToyProblemDemo extends App {

  val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv")) //(0 to 40, ::)
  val x = data(::, 0).toDenseMatrix.t
  val y = data(::, 1 to 2)

  val z = x(0 until x.rows by 10, ::)

  val now = System.currentTimeMillis()
  val initialToyModel = createCogpModel(x, y, z)
  val trainedToyModel = cogpTrain(x, y, initialToyModel, iterNum = 500)
  println(System.currentTimeMillis() - now)

  val predictedY = cogpPredict(x, trainedToyModel)
  val predictedOutput0 = predictedY(::, 0).map(_.m)
  val predictedOutput1 = predictedY(::, 1).map(_.m)

  plotPredictions()

  // val figure = Figure()
  //   figure.subplot(0).legend = true

  println(trainedToyModel.h(1).u.v)

  //figure.subplot(0) += plot(trainedToyModel.g(0).z(::,0),trainedToyModel.g(0).u.m)
  //   figure.subplot(0) += plot(trainedToyModel.h(0).z(::,0),trainedToyModel.h(1).u.m)

  private def plotPredictions() = {
    val figure = Figure()
    figure.subplot(0).legend = true

    figure.subplot(0) += plot(x(::, 0), y(::, 0), name = "actual output 0")
    figure.subplot(0) += plot(x(::, 0), predictedOutput0, name = "predicted output 0")

    figure.subplot(0) += plot(x(::, 0), y(::, 1), name = "actual output 1")
    figure.subplot(0) += plot(x(::, 0), predictedOutput1, name = "predicted output 1")

  }
}