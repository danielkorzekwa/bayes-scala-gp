package dk.gp.hgpr

import scala.util.Random

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.numerics.pow
import breeze.stats.mean
import breeze.stats.mean.reduce_Double
import dk.bayes.math.accuracy.rmse
import dk.gp.hgpr.util.calcHgprRMSEandR2
import breeze.linalg._
import breeze.numerics._

object accuracy {

  def apply(model: HgprModel): HgprAccuracy = {

    val (allRMSE, allR2) = calcRMSEAndR2(model.x, model.y, model)

    val (looCVn,looCVRMSE, looCVR2) = calcHgprRMSEandR2(model)

    val (trainingSet, testSet) = split(DenseMatrix.horzcat(model.x, model.y.toDenseMatrix.t))

    val trainingSetX = trainingSet(::, (0 until trainingSet.cols - 1))
    val trainingSetY = trainingSet(::, trainingSet.cols - 1)
    val trainingSetModel = model.copy(x = trainingSetX, y = trainingSetY)

    val (trainRMSE, trainR2) = calcRMSEAndR2(trainingSetX, trainingSetY, trainingSetModel)

    val testSetX = testSet(::, (0 until testSet.cols - 1))
    val testSetY = testSet(::, testSet.cols - 1)

    val (testRMSE, testR2) = calcRMSEAndR2(testSetX, testSetY, trainingSetModel)

    HgprAccuracy(
      model.y.size, allRMSE, allR2,
      looCVRMSE, looCVR2,
      trainingSetY.size, trainRMSE, trainR2,
      testSetY.size, testRMSE, testR2)
  }

  private def calcRMSEAndR2(x: DenseMatrix[Double], y: DenseVector[Double], model: HgprModel): (Double, Double) = {

    val predicted = hgprPredict(x, model).map(x => x.m)
    val modelRMSE = rmse(predicted, y)

    val predictedBaseline = DenseVector.fill(y.size)(mean(model.y))
    val baselineRMSE = rmse(predictedBaseline, y)

    val r2 = 1 - pow(modelRMSE, 2) / pow(baselineRMSE, 2)

    (modelRMSE, r2)
  }

  private def split(data: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {

    val random = new Random(54354)
    val (trainingSetIdx, testSetIdx) = (0 until data.rows).partition { x => random.nextDouble() < 0.7 }

    val trainingSet = data(trainingSetIdx, ::).toDenseMatrix
    val testSet = data(testSetIdx, ::).toDenseMatrix

    (trainingSet, testSet)
  }
}