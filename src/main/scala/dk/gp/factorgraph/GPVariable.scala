package dk.gp.factorgraph

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.bayes.factorgraph2.factor.CanonicalGaussianFactor
import dk.bayes.factorgraph2.factor.CanonicalLinearGaussianFactor
import dk.bayes.factorgraph2.variable.CanonicalGaussianVariable
import dk.bayes.math.gaussian.MultivariateGaussian
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.gp.cov.CovFunc
import dk.gp.gp.ConditionalGPFactory
import dk.gp.gp.GPPredictSingle

class GPVariable(name: String = "na", val x: DenseMatrix[Double], val covFunc: CovFunc, val covFuncParams: DenseVector[Double], val gpMean: Double) extends CanonicalGaussianVariable(name) {

  def toGP(): GPPredictSingle = {
    val variableValCanon = get().asInstanceOf[DenseCanonicalGaussian]
    val variableVal = MultivariateGaussian(variableValCanon.mean, variableValCanon.variance)
    GPPredictSingle(variableVal, x, covFunc, covFuncParams, gpMean)
  }

  def createFactor(): CanonicalGaussianFactor = {
    val m = DenseVector.fill[Double](x.rows)(gpMean)
    val v = covFunc.cov(x, x, covFuncParams) + DenseMatrix.eye[Double](x.rows) * 1e-7

    CanonicalGaussianFactor(this, m, v)
  }

  def createFactor(parentVariable: GPVariable): CanonicalLinearGaussianFactor = {
    require(parentVariable.covFunc.equals(covFunc))
    require(parentVariable.covFuncParams.equals(covFuncParams))
    require(parentVariable.gpMean.equals(gpMean))

    val (a, b, v) = ConditionalGPFactory(parentVariable.x, covFunc, covFuncParams, gpMean).create(x)

    CanonicalLinearGaussianFactor(parentVariable, this, a, b, v)
  }

}