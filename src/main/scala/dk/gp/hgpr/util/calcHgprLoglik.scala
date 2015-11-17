package dk.gp.hgpr.util

import breeze.linalg.DenseMatrix
import dk.gp.gpr.gprLoglik
import breeze.linalg.cholesky
import dk.gp.math.invchol
import breeze.numerics._
import dk.gp.math.MultivariateGaussian
import dk.gp.math.MultivariateGaussian
import dk.gp.hgpr.HgprModel
import dk.gp.gp.gpPredictSingle
import dk.gp.math.MultivariateGaussian

object calcHgprLoglik {

  def apply(model: HgprModel): Double = {

    val hgpFactorGraph = HgprFactorGraph(model.x, model.y, model.u, model.covFunc, model.covFuncParams, model.likNoiseLogStdDev)
    val uPosterior = hgpFactorGraph.calcUPosterior()

    val taskIds = model.x(::, 0).toArray.distinct

    val logliks = taskIds.map { taskId =>

      val idx = model.x(::, 0).findAll { x => x == taskId }
      val taskX = model.x(idx, ::).toDenseMatrix
      val taskY = model.y(idx).toDenseVector

      val xFactorMsgUp = hgpFactorGraph.getXFactorMsgUp(taskId.toInt)

      val uVarMsgDown = uPosterior / xFactorMsgUp
      val xPrior = gpPredictSingle(taskX, MultivariateGaussian(uVarMsgDown.mean,uVarMsgDown.variance), model.u,model.covFunc, model.covFuncParams)
      val cPriorVarWithNoise = xPrior.v + DenseMatrix.eye[Double](taskX.rows) * exp(2 * model.likNoiseLogStdDev)
      val loglik = gprLoglik(xPrior.m, cPriorVarWithNoise, invchol(cholesky(cPriorVarWithNoise).t), taskY)

      loglik
    }

    val totalLoglik = logliks.sum

    totalLoglik
  }
}