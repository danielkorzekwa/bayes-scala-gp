package dk.gp.cov

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.math.sqDist
import breeze.numerics._

/**
 * Implementation based 'http://www.gaussianprocess.org/gpml/code/matlab/doc/index.html'
 *
 *  Squared Exponential covariance function with isotropic distance measure. The
 * covariance function is parameterized as:
 *
 * k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
 *
 * where the P matrix is ell^2 times the unit matrix and sf^2 is the signal
 * variance.
 *
 * Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
 *
 * @param sf - log of signal standard deviation
 * @param ell - log of length scale standard deviation
 */

case class CovSEiso() extends CovFunc {

  def cov(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): DenseMatrix[Double] = {

    val logSf = covFuncParams(0)
    val logEll = covFuncParams(1)
    val ell = exp(logEll)

    val sqDistMatrix = sqDist(x1.t / ell, x2.t / ell)
    val covMatrix = exp(2 * logSf) * exp(-0.5 * sqDistMatrix)
    covMatrix
  }

  def covD(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): Array[DenseMatrix[Double]] = {

    val logSf = covFuncParams(0)
    val logEll = covFuncParams(1)
    val ell = exp(logEll)

    val sqDistMatrix = sqDist(x1.t / ell, x2.t / ell)

    val expSqDistMatrix = exp(-0.5 * sqDistMatrix)

    val covMatrixDSf = 2 * exp(2 * logSf) * expSqDistMatrix
    val covMatrixDEll = exp(2 * logSf) * expSqDistMatrix :* sqDistMatrix

    Array(covMatrixDSf, covMatrixDEll)
  }

}