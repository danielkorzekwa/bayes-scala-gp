package dk.gp.gpc.factorgraph2

import dk.bayes.math.gaussian.canonical.CanonicalGaussian
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.gp.gpPredictSingle
import dk.bayes.dsl._

/**
 * Linear Conditional Gaussian p(t|x) = N(t|Ax+b,v)
 */
case class LinearGaussianFactor(v1: GaussianVariable, v2: GaussianVariable, a: DenseMatrix[Double], b: DenseVector[Double], v: DenseMatrix[Double]) extends DoubleFactor[CanonicalGaussian, CanonicalGaussian] {

  def getV1(): Variable[CanonicalGaussian] = v1
  def getV2(): Variable[CanonicalGaussian] = v2

  def getInitialMsgV1(): CanonicalGaussian = {
    val m = DenseVector.zeros[Double](a.cols)
    val v = DenseMatrix.eye[Double](a.cols) * 1000d
    DenseCanonicalGaussian(m, v)
  }

  def calcNewMsgV1(): CanonicalGaussian = {

    val v2 = this.getV2.get.asInstanceOf[DenseCanonicalGaussian]
    val oldFactorMsgUp = this.getMsgV2.get.asInstanceOf[DenseCanonicalGaussian]
    val taskXVarMsgUp = v2 / oldFactorMsgUp

    val xFactorCanon = DenseCanonicalGaussian(a, b, v)

    val factorTimesMsg = xFactorCanon * taskXVarMsgUp.extend(a.cols + a.rows, a.cols)
    val newXFactorMsgUp = factorTimesMsg.marginal((0 until a.cols): _*)

    newXFactorMsgUp
  }

  def getInitialMsgV2(): CanonicalGaussian = {
    val m = DenseVector.zeros[Double](a.rows)
    val v = DenseMatrix.eye[Double](a.rows) * 1000d
    DenseCanonicalGaussian(m, v)
  }

  def calcNewMsgV2(): CanonicalGaussian = {
    val v1 = this.getV1.get.asInstanceOf[DenseCanonicalGaussian]
    val oldFactorMsgUp = this.getMsgV1.get.asInstanceOf[DenseCanonicalGaussian]

    val uVarMsgDown = v1 / oldFactorMsgUp

    val fVariable = dk.bayes.dsl.variable.gaussian.multivariate.MultivariateGaussian(uVarMsgDown.mean, uVarMsgDown.variance)

    val yVariable = dk.bayes.dsl.variable.Gaussian(a, fVariable, b, v)
    val yPosterior = infer(yVariable)

    DenseCanonicalGaussian(yPosterior.m, yPosterior.v)
  }

}