package dk.gp.hgpc.util

import dk.gp.hgpc.HgpcModel
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.bayes.dsl.factor.DoubleFactor
import dk.bayes.math.gaussian.canonical.CanonicalGaussian
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.bayes.math.gaussian.canonical.SparseCanonicalGaussian
import breeze.numerics._

object calcHGPCLoglik {

  def apply(model: HgpcModel): Double = {

    val hgpcFactorGraph = HgpcFactorGraph(model)
    val (calib, iters) = calibrateHgpcFactorGraph(hgpcFactorGraph, maxIter = 10)

    apply(hgpcFactorGraph)

  }

  def apply(calibratedHgpcFactorGraph: HgpcFactorGraph): Double = {

    val totalLoglik = calibratedHgpcFactorGraph.taskIds.map { taskId =>

      calibratedHgpcFactorGraph.taskYFactorsMap(taskId).map { taskYFactor =>
        val outcome1Prob = taskYFactor.calcNewMsgV2()
        if (taskYFactor.v2.k == 1) log(outcome1Prob) else log1p(-outcome1Prob)
      }.sum

    }.sum
    totalLoglik
  }
  
  /**
   * This function will update calibratedHgpcFactorGraph for the purpose of computing approximated loglikelihood of evidence given provided covFuncParams and gpMean parameters
   * This might be used for computing approximated derivatives of  approximated loglikelihood(lowerBound) for covFuncParams and gpMean parameters
   */
   def apply(calibratedHgpcFactorGraph: HgpcFactorGraph,covFuncParams: DenseVector[Double], gpMean: Double): Double = {
     ???
   }
}