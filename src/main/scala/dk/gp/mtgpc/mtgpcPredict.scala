package dk.gp.mtgpc

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.gpc.GpcModel
import dk.gp.gpc.gpcPredict
import breeze.stats._
object mtgpcPredict {

  /**
   * Returns vector of probabilities for a class 1
   */
  def apply(t: DenseMatrix[Double], model: MtgpcModel): DenseVector[Double] = {

    val predictedVector = DenseVector.zeros[Double](t.rows)

    val testTaskIds = t(::, 0).toArray.map(_.toInt).distinct

    lazy val yMean = mean(model.y.map(y => if (y == 1) 1d else 0))

    testTaskIds.foreach { taskId =>

      val trainTaskIdx = model.x(::, 0).findAll { x => x == taskId }

      val taskPredictIdx = t(::, 0).findAll { x => x == taskId }

      val taskPredicted = if (trainTaskIdx.size == 0) DenseVector.fill(taskPredictIdx.size)(yMean)
      else {

        val trainTaskX = model.x(trainTaskIdx, ::).toDenseMatrix
        val trainTaskY = model.y(trainTaskIdx).toDenseVector
        val taskGpcModel = GpcModel(trainTaskX, trainTaskY, model.covFunc, model.covFuncParams, model.gpMean)

        val taskPredictX = t(taskPredictIdx, ::).toDenseMatrix
        val taskPredicted = gpcPredict(taskPredictX, taskGpcModel)
        taskPredicted
      }

      predictedVector(taskPredictIdx) := taskPredicted
    }

    predictedVector
  }

}