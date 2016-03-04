package dk.gp.hgpc

import breeze.linalg.DenseMatrix
import scala.util.Random
import dk.bayes.math.accuracy.loglik
import dk.bayes.math.accuracy.binaryAcc
import breeze.linalg.DenseVector
import breeze.numerics._
import breeze.stats._
import dk.gp.hgpc.util.calcHGPCLoglik
import dk.gp.hgpc.util.calcHGPCAcc

object accuracy {

  def apply(model: HgpcModel): String = {

    val (allLoglik, allAvgLoglik, allAcc,allTpr,allTnr) = calcLoglikAndAcc(model.x, model.y, model)

    val looCVLoglik = calcHGPCLoglik(model)
    val (looAcc,looTpr,looTnr) = calcHGPCAcc(model)

    val (trainingSet, testSet) = split(DenseMatrix.horzcat(model.x, model.y.toDenseMatrix.t))

    val trainingSetX = trainingSet(::, (0 until trainingSet.cols - 1))
    val trainingSetY = trainingSet(::, trainingSet.cols - 1)
    val trainingSetModel = model.copy(x = trainingSetX, y = trainingSetY)

    val (trainLoglik, trainAvgLoglik, trainAcc,trainTpr,trainTnr) = calcLoglikAndAcc(trainingSetX, trainingSetY, trainingSetModel)

    val testSetX = testSet(::, (0 until testSet.cols - 1))
    val testSetY = testSet(::, testSet.cols - 1)

    val (testLoglik, testAvgLoglik, testAcc,testTpr,testTnr) = calcLoglikAndAcc(testSetX, testSetY, trainingSetModel)

    val allReport = "All: n=%2d, loglik=%.2f, avgLoglik=%.2f, acc=%.3f, tpr=%.3f, tnr=%.3f".format(model.y.size, allLoglik, allAvgLoglik, allAcc,allTpr,allTnr)

    val looCVReport = "LooCV: loglik=%.2f, avgLoglik=%.2f,acc=%.3f, tpr=%.3f, tnr=%.3f".format(looCVLoglik, looCVLoglik / model.y.size, looAcc,looTpr,looTnr)

    val trainReport = "Train: n=%2d, loglik=%.2f, avgLoglik=%.2f, acc=%.3f, tpr=%.3f, tnr=%.3f".format(trainingSetY.size, trainLoglik, trainAvgLoglik, trainAcc,trainTpr,trainTnr)

    val testReport = "Test: n=%2d, loglik=%.2f, avgLoglik=%.2f, acc=%.3f, tpr=%.3f, tnr=%.3f".format(testSetY.size, testLoglik, testAvgLoglik, testAcc,testTpr,testTnr)

    allReport + "\n" + looCVReport + "\n" + trainReport + "\n" + testReport
  }

  /**
   * @return (loglik,avgLoglik,acc,tpr,tnr)
   */
  private def calcLoglikAndAcc(x: DenseMatrix[Double], y: DenseVector[Double], model: HgpcModel): Tuple5[Double, Double, Double, Double, Double] = {

    val predicted = hgpcPredict(x, model)

    val modelLoglik = loglik(predicted, y)
    val avgLoglik = modelLoglik / y.size
    val acc = binaryAcc(predicted, y)
    val tpr = binaryAcc(predicted(y :== 1d).toDenseVector, y(y :== 1d).toDenseVector)
    val tnr = binaryAcc(predicted(y :== 0d).toDenseVector, y(y :== 0d).toDenseVector)

    (modelLoglik, avgLoglik, acc, tpr, tnr)

  }

  private def split(data: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {

    val random = new Random(54354)
    val (trainingSetIdx, testSetIdx) = (0 until data.rows).partition { x => random.nextDouble() < 0.7 }

    val trainingSet = data(trainingSetIdx, ::).toDenseMatrix
    val testSet = data(testSetIdx, ::).toDenseMatrix

    (trainingSet, testSet)
  }
}