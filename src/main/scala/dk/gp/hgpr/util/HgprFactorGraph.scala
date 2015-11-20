package dk.gp.hgpr.util

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.cholesky
import dk.gp.cov.CovFunc
import dk.gp.math.invchol
import breeze.linalg.inv
import breeze.numerics._
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.bayes.dsl.variable.Gaussian
import dk.bayes.dsl.infer
import dk.gp.gp.ConditionalGPFactory

case class HgprFactorGraph(x: DenseMatrix[Double], y: DenseVector[Double], u: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], likNoiseLogStdDev: Double) {

  private val kUU = covFunc.cov(u, u, covFuncParams) + DenseMatrix.eye[Double](u.rows) * 1e-7
  private val uMean = DenseVector.zeros[Double](u.rows)
  private val priorU = DenseCanonicalGaussian(uMean, kUU)

  private val condGPFactory = ConditionalGPFactory(u, covFunc, covFuncParams, mean = 0)

  private val xFactorMsgUpByTaskId: Map[Int, DenseCanonicalGaussian] = {

    val taskIds = x(::, 0).toArray.distinct

    val taskMsgsUp = taskIds.par.map { taskId =>
      val idx = x(::, 0).findAll { x => x == taskId }
      val taskX = x(idx, ::).toDenseMatrix
      val taskY = y(idx).toDenseVector

      val (a, b, v) = condGPFactory.create(taskX)

      val vWithNoise = v + DenseMatrix.eye[Double](taskX.rows) * exp(2 * likNoiseLogStdDev)
      val priorUVariable = dk.bayes.dsl.variable.Gaussian(priorU.mean, priorU.variance)
      val xVariable = Gaussian(a, priorUVariable, b, vWithNoise, yValue = taskY)
      val uPosterior = infer(priorUVariable)
      val xMsgUp = DenseCanonicalGaussian(uPosterior.m, uPosterior.v) / priorU

      taskId.toInt -> xMsgUp

    }

    taskMsgsUp.toList.toMap
  }

  def getUFactorMsgDown(): DenseCanonicalGaussian = priorU

  def getXFactorMsgUp(cId: Int): DenseCanonicalGaussian = xFactorMsgUpByTaskId(cId)
  def getXFactorMsgs(): Seq[DenseCanonicalGaussian] = xFactorMsgUpByTaskId.values.toList

  def calcUPosterior(): DenseCanonicalGaussian = getUFactorMsgDown() * getXFactorMsgs.reduceLeft(_ * _)

}
