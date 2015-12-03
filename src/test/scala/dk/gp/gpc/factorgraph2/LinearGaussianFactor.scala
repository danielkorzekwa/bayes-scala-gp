package dk.gp.gpc.factorgraph2

import dk.bayes.math.gaussian.canonical.CanonicalGaussian
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

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
    ???
  }
  
   def getInitialMsgV2(): CanonicalGaussian = {
    val m = DenseVector.zeros[Double](a.rows)
    val v = DenseMatrix.eye[Double](a.rows) * 1000d
    DenseCanonicalGaussian(m, v)
  }

  def calcNewMsgV2(): CanonicalGaussian = {
    ???
  }

}