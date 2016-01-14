package dk.gp.gpc

import breeze.linalg.DenseMatrix
import dk.gp.cov.CovFunc
import breeze.linalg.DenseVector

/**
 * @param x
 * @param y vector of {-1,1}
 * @param covFunc
 * @param covFuncParams
 * @param mean
 */
case class GpcModel(x: DenseMatrix[Double], y: DenseVector[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], gpMean: Double = 0)