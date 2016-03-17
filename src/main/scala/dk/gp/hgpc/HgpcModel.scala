package dk.gp.hgpc

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc

/**
 * @param x [taskId, feature1, feature2, ...]
 * @param y vector of {-1,1}
 * @param u Inducing variables for a parent GP [taskId, feature1, feature2, ...]
 * @param covFunc
 * @param covFuncParams
 * @param mean
 */
case class HgpcModel(x: DenseMatrix[Double], y: DenseVector[Double], u: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], mean: Double) {

  def getTaskData(taskId: Int): Tuple2[DenseMatrix[Double], DenseVector[Double]] = {

    val taskIdx = x(::, 0).findAll { tId => tId == taskId }
    val taskX = x(taskIdx, ::).toDenseMatrix
    val taskY = y(taskIdx).toDenseVector

    (taskX, taskY)
  }
}