package dk.gp.cov

import scala.math._
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

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

    val sf = covFuncParams(0)
    val ell = covFuncParams(1)

    def op(v1: DenseVector[Double], v2: DenseVector[Double]): Double = cov(v1.toArray, v2.toArray, sf, ell)
    val covMatrix = covFunc(x1, x2, op)
    covMatrix
  }

  def cov(x1: Array[Double], x2: Array[Double], sf: Double, ell: Double): Double = {

    require(x1.size == x2.size, "Vectors x1 and x2 have different sizes")
    val expArg = -0.5 * distance(x1, x2, exp(2 * ell))
    exp(2 * sf) * exp(expArg)
  }

  private def distance(x1: Array[Double], x2: Array[Double], l: Double): Double = {

    var distance = 0d
    var i = 0

    while (i < x1.size) {
      distance += pow(x1(i) - x2(i), 2) / l
      i += 1
    }

    distance
  }

   //@TODO delete this method
  def covD(x: DenseMatrix[Double], covFuncParams: DenseVector[Double]): Array[DenseMatrix[Double]] = {

    val sf = covFuncParams(0)
    val ell = covFuncParams(1)

    def dfDsf(v1: DenseVector[Double], v2: DenseVector[Double]): Double = df_dSf(v1.toArray, v2.toArray, sf, ell)
    val covMatrixDSf = covFunc(x, x, dfDsf)

    def dfDEll(v1: DenseVector[Double], v2: DenseVector[Double]): Double = df_dEll(v1.toArray, v2.toArray, sf, ell)
    val covMatrixDEll = covFunc(x, x, dfDEll)

    Array(covMatrixDSf, covMatrixDEll)
  }
  
   def covD(x1: DenseMatrix[Double],x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): Array[DenseMatrix[Double]] = {

    val sf = covFuncParams(0)
    val ell = covFuncParams(1)

    def dfDsf(v1: DenseVector[Double], v2: DenseVector[Double]): Double = df_dSf(v1.toArray, v2.toArray, sf, ell)
    val covMatrixDSf = covFunc(x1, x2, dfDsf)

    def dfDEll(v1: DenseVector[Double], v2: DenseVector[Double]): Double = df_dEll(v1.toArray, v2.toArray, sf, ell)
    val covMatrixDEll = covFunc(x1, x2, dfDEll)

    Array(covMatrixDSf, covMatrixDEll)
  }

  def df_dSf(x1: Array[Double], x2: Array[Double], sf: Double, ell: Double): Double = {
    require(x1.size == x2.size, "Vectors x1 and x2 have different sizes")

    val expArg = -0.5 * distance(x1, x2, exp(2 * ell))
    2 * exp(2 * sf) * exp(expArg)
  }

  def df_dEll(x1: Array[Double], x2: Array[Double], sf: Double, ell: Double): Double = {
    require(x1.size == x2.size, "Vectors x1 and x2 have different sizes")

    val dfDEll = if (x1.size == 1 && x2.size == 1 && x1(0) == x2(0)) 0
    else {
      val expArg = -0.5 * distance(x1, x2, exp(2 * ell))
      val d = -0.5 * distance(x1, x2, exp(2 * ell) / (-2d))

      exp(2 * sf) * exp(expArg) * d
    }
    dfDEll
  }

  private def covFunc(x1: DenseMatrix[Double], x2: DenseMatrix[Double], op: (DenseVector[Double], DenseVector[Double]) => Double): DenseMatrix[Double] = {
    val matrix = DenseMatrix.zeros[Double](x1.rows, x2.rows)

    for (rowIndex <- 0 until x1.rows) {

      val x1Val = x1(rowIndex, ::).t

      for (colIndex <- 0 until x2.rows) {
        val x2Val = x2(colIndex, ::).t

        matrix(rowIndex, colIndex) = op(x1Val, x2Val)
      }
    }

    matrix
  }
}