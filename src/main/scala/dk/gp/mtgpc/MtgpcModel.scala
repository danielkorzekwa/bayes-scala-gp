package dk.gp.mtgpc

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc

/**
 * @param x [taskId, feature1, feature2, ...]
 * @param y vector of {-1,1}
 * @param covFunc
 * @param covFuncParams
 * @param gpMean
 */
case class MtgpcModel(x: DenseMatrix[Double], y: DenseVector[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], gpMean: Double)
