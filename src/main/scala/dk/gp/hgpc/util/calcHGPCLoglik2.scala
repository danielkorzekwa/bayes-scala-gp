package dk.gp.hgpc.util

import dk.gp.hgpc.HgpcModel
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.bayes.dsl.factor.DoubleFactor
import dk.bayes.math.gaussian.canonical.CanonicalGaussian
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.bayes.math.gaussian.canonical.SparseCanonicalGaussian
import breeze.numerics._

object calcHGPCLoglik2 {

  def apply(model: HgpcModel): Double = {

    val hgpcFactorGraph = HgpcFactorGraph2(model)
    val (calib, iters) = calibrateHgpcFactorGraph2(hgpcFactorGraph, maxIter = 10)

    val totalLoglik = hgpcFactorGraph.taskIds.map { taskId =>

      hgpcFactorGraph.taskYFactorsMap(taskId).map { taskYFactor =>
        val outcome1Prob = taskYFactor.calcNewMsgV2()
        if (taskYFactor.v2.k == 1) log(outcome1Prob) else log1p(-outcome1Prob)
      }.sum

    }.sum
    totalLoglik
  }
}