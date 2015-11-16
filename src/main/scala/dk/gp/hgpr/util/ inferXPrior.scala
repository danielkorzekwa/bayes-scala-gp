package dk.gp.hgpr.util

import dk.gp.cov.CovFunc
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import dk.gp.math.invchol
import breeze.linalg.cholesky
import breeze.linalg.inv
import breeze.numerics._
import dk.gp.math.UnivariateGaussian
import dk.gp.math.MultivariateGaussian
import dk.bayes.dsl.infer
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.bayes.dsl.variable.Gaussian

object inferXPrior {

  def apply(x: DenseMatrix[Double], u: DenseMatrix[Double], uPosterior: DenseCanonicalGaussian, covFunc: CovFunc, covFuncParams: DenseVector[Double], likNoiseLogStdDev: Double): (DenseVector[Double], DenseMatrix[Double]) = {

    val kUU = covFunc.cov(u, u, covFuncParams) + DenseMatrix.eye[Double](u.rows) * 1e-7

    val kXU = covFunc.cov(x, u, covFuncParams)
    val kXX = covFunc.cov(x, x, covFuncParams) + DenseMatrix.eye[Double](x.rows) * 1e-7

    val A = kXU * invchol(cholesky(kUU).t)
    val b = DenseVector.zeros[Double](x.rows)
    val kXUInvLUU = kXU * inv(cholesky(kUU).t)
    val v = kXX - kXUInvLUU * kXUInvLUU.t

    val uPosteriorCopy = dk.bayes.dsl.variable.gaussian.multivariate.MultivariateGaussian(uPosterior.mean, uPosterior.variance)

    val cVariable = Gaussian(A, uPosteriorCopy, b, v)
    val cPosterior = infer(cVariable)

    (cPosterior.m, cPosterior.v)

  }
}