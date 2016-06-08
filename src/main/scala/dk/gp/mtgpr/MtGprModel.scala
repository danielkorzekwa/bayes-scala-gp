package dk.gp.mtgpr

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc

/**
 * @param data Training data [x matrix y vector] for n tasks
 */
case class MtGprModel(data: Seq[DenseMatrix[Double]], covFunc: CovFunc, covFuncParams: DenseVector[Double], likNoiseLogStdDev: Double) {

}

object MtGprModel {

  /**
   * @param x [taskId, feature1, feature2,...]
   * @param y
   * @param covFunc
   * @param initialCovFuncParams
   * @param initialLikNoiseLogStdDev
   */
  def apply(x: DenseMatrix[Double], y: DenseVector[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], likNoiseLogStdDev: Double): MtGprModel = {
    val taskIds = x(::, 0).toArray.toList.distinct

    val data = taskIds.map { cId =>
      val idx = x(::, 0).findAll { x => x == cId }
      val taskX = x(idx, ::).toDenseMatrix
      val taskY = y(idx).toDenseVector
      val taskXY = DenseMatrix.horzcat(taskX, taskY.toDenseMatrix.t)

      taskXY
    }

    MtGprModel(data, covFunc, covFuncParams, likNoiseLogStdDev)
  }
}