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

  def apply(model: HgpcModel): Double = {

    val hgpcFactorGraph = HgpcFactorGraph(model)
    val (calib, iters) = calibrateHgpcFactorGraph(hgpcFactorGraph, maxIter = 10)

    apply(hgpcFactorGraph)

  }

  def apply(calibratedHgpcFactorGraph: HgpcFactorGraph): Double = {

    val predictedVSActual = calibratedHgpcFactorGraph.taskIds.flatMap { taskId =>

      calibratedHgpcFactorGraph.taskYFactorsMap(taskId).map { taskYFactor =>
        val outcome1Prob = taskYFactor.calcNewMsgV2()
        DenseVector(outcome1Prob, taskYFactor.v2.k)
      }

    }

    val predictedVSActualMatrix = DenseVector.horzcat(predictedVSActual: _*).t
    val predicted = predictedVSActualMatrix(::, 0)
    val actual = predictedVSActualMatrix(::, 1)

    binaryAcc(predicted, actual)
  }

}