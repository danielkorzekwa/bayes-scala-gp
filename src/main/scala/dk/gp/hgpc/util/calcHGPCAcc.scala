package dk.gp.hgpc.util

import dk.gp.hgpc.HgpcModel
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.bayes.dsl.factor.DoubleFactor
import dk.bayes.math.gaussian.canonical.CanonicalGaussian
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.bayes.math.gaussian.canonical.SparseCanonicalGaussian
import breeze.numerics._
import dk.bayes.math.accuracy.loglik
import dk.bayes.math.accuracy.binaryAcc

object calcHGPCAcc {

  /**
   * @return (acc,tpr,tnr)
   */
  def apply(model: HgpcModel): Tuple3[Double, Double, Double] = {

    val hgpcFactorGraph = HgpcFactorGraph(model)
    val (calib, iters) = calibrateHgpcFactorGraph(hgpcFactorGraph, maxIter = 10)

    apply(hgpcFactorGraph)

  }

  /**
   * @return (acc,tpr,tnr)
   */
  def apply(calibratedHgpcFactorGraph: HgpcFactorGraph): Tuple3[Double, Double, Double] = {

    val predictedVSActual = calibratedHgpcFactorGraph.taskIds.flatMap { taskId =>

      calibratedHgpcFactorGraph.taskYFactorsMap(taskId).map { taskYFactor =>
        val outcome1Prob = taskYFactor.calcNewMsgV2()
        DenseVector(outcome1Prob, taskYFactor.v2.k)
      }

    }

    val predictedVSActualMatrix = DenseVector.horzcat(predictedVSActual: _*).t
    val predicted = predictedVSActualMatrix(::, 0)
    val actual = predictedVSActualMatrix(::, 1)

    val acc = binaryAcc(predicted, actual)
    val tpr = binaryAcc(predicted(actual :== 1d).toDenseVector, actual(actual :== 1d).toDenseVector)
    val tnr = binaryAcc(predicted(actual :== 0d).toDenseVector, actual(actual :== 0d).toDenseVector)

    (acc, tpr, tnr)
  }

}