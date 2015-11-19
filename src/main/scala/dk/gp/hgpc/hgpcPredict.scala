package dk.gp.hgpc

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.math.UnivariateGaussian
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.gp.math.MultivariateGaussian
import dk.gp.gp.gpPredictSingle
import dk.bayes.math.gaussian.Gaussian
import breeze.numerics._

/**
 * Hierarchical Gaussian Process classification. Multiple Gaussian Processes for n tasks with a single shared parent GP.
 */
object hgpcPredict {

  case class TaskPosterior(x: DenseMatrix[Double], xPosterior: DenseCanonicalGaussian)

  /**
   * Returns vector of probabilities for a class 1
   */
  def apply(t: DenseMatrix[Double], model: HgpcModel): DenseVector[Double] = {
    val taskPosteriorByTaskId: Map[Int, TaskPosterior] = createTaskPosteriorByTaskId(t, model)

    val predictedArray = (0 until t.rows).par.map { rowIndex =>

      val tRow = t(rowIndex, ::).t
      val taskId = tRow(0).toInt
      val taskPosterior = taskPosteriorByTaskId(taskId)

      val tTestPrior = gpPredictSingle(tRow.toDenseMatrix, MultivariateGaussian(taskPosterior.xPosterior.mean, taskPosterior.xPosterior.variance), taskPosterior.x, model.covFunc, model.covFuncParams, model.mean)
      val predictedProb = Gaussian.stdCdf(tTestPrior.m(0) / sqrt(1d + tTestPrior.v(0, 0)))

      predictedProb
    }.toArray
    DenseVector(predictedArray)
  }

  private def createTaskPosteriorByTaskId(xTest: DenseMatrix[Double], model: HgpcModel): Map[Int, TaskPosterior] = {
    throw new UnsupportedOperationException("Not implemented yet")
  }
}