package dk.gp.gpc.factorgraph2

import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.bayes.math.gaussian.canonical.CanonicalGaussian
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

case class MultivariateGaussianFactor(v1: GaussianVariable, m: DenseVector[Double], v: DenseMatrix[Double]) extends SingleFactor[CanonicalGaussian] {

  def getV1(): Variable[CanonicalGaussian] = v1

  def getInitialMsgV1(): CanonicalGaussian = DenseCanonicalGaussian(m, v)

  def calcNewMsgV1(): CanonicalGaussian = DenseCanonicalGaussian(m, v)
}