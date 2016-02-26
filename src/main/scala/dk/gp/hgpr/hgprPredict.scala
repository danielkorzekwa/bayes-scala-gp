package dk.gp.hgpr

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.numerics.exp
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.gp.hgpr.util.HgprFactorGraph
import dk.gp.gp.GPPredictSingle
import dk.bayes.math.gaussian.Gaussian
import dk.bayes.math.gaussian.MultivariateGaussian

/**
 * Hierarchical Gaussian Process regression. Multiple Gaussian Processes for n tasks with a single shared parent GP.
 */
object hgprPredict {

  case class TaskPosterior(x: DenseMatrix[Double], xPosterior: DenseCanonicalGaussian)

  def apply(xTest: DenseMatrix[Double], model: HgprModel): DenseVector[Gaussian] = {

    val taskPosteriorByTaskId: Map[Int, TaskPosterior] = createTaskPosteriorByTaskId(xTest, model)

    val predictedArray = (0 until xTest.rows).par.map { rowIndex =>

      val xRow = xTest(rowIndex, ::).t
      val taskId = xRow(0).toInt
      val taskPosterior = taskPosteriorByTaskId(taskId)

      val xTestPrior = GPPredictSingle(MultivariateGaussian(taskPosterior.xPosterior.mean, taskPosterior.xPosterior.variance), taskPosterior.x, model.covFunc, model.covFuncParams).predictSingle(xRow.toDenseMatrix)
      Gaussian(xTestPrior.m(0), xTestPrior.v(0, 0))
    }.toArray
    DenseVector(predictedArray)

  }

  private def createTaskPosteriorByTaskId(xTest: DenseMatrix[Double], model: HgprModel): Map[Int, TaskPosterior] = {

    val hgpFactorGraph = HgprFactorGraph(model.x, model.y, model.u, model.covFunc, model.covFuncParams, model.likNoiseLogStdDev)
    val uPosterior = hgpFactorGraph.calcUPosterior()

    val taskIds = xTest(::, 0).toArray.distinct

    val taskPosteriorByTaskId: Map[Int, TaskPosterior] = taskIds.map { taskId =>

      val idx = model.x(::, 0).findAll { x => x == taskId }
      val taskX = model.x(idx, ::).toDenseMatrix
      val taskY = model.y(idx).toDenseVector

      val taskPosterior = if (taskY.size == 0) TaskPosterior(model.u, uPosterior)
      else {
        val taskXTestIdx = xTest(::, 0).findAll(x => x == taskId)
        val taskXTest = xTest(taskXTestIdx, ::).toDenseMatrix

        val taskXX = DenseMatrix.vertcat(taskX, taskXTest)
        val xPrior = GPPredictSingle(MultivariateGaussian(uPosterior.mean, uPosterior.variance), model.u, model.covFunc, model.covFuncParams).predictSingle(taskXX)

        val xPriorVariable = dk.bayes.dsl.variable.gaussian.multivariate.MultivariateGaussian(xPrior.m, xPrior.v)

        val A = DenseMatrix.horzcat(DenseMatrix.eye[Double](taskX.rows), DenseMatrix.zeros[Double](taskX.rows, taskXTest.rows))
        val yVar = DenseMatrix.eye[Double](taskY.size) * exp(2d * model.likNoiseLogStdDev)
        val yVariable = dk.bayes.dsl.variable.Gaussian(A, xPriorVariable, b = DenseVector.zeros[Double](taskX.rows), yVar, yValue = taskY) //y variable

        val xPosterior = dk.bayes.dsl.infer(xPriorVariable)
        TaskPosterior(taskXX, DenseCanonicalGaussian(xPosterior.m, xPosterior.v))
      }
      // @TODO Simple impl if xTest is in testX - use it in this situation
      //      val (xPriorMean, cPriorVar) = inferXPrior(testX, model.u, uPosterior, model.covFunc, model.covFuncParams, model.likNoiseLogStdDev)
      //      val xPriorVariable = dk.bayes.dsl.variable.gaussian.multivariate.MultivariateGaussian(xPriorMean, cPriorVar)
      //
      //      val yVar = DenseMatrix.eye[Double](testY.size) * exp(2d * model.likNoiseLogStdDev)
      //      val yVariable = Gaussian(xPriorVariable, yVar, yValue = testY) //y variable
      //
      //      val xPosterior = dk.bayes.dsl.infer(xPriorVariable)
      //
      //      val xTestPrior = inferXPrior(xRow.toDenseMatrix, testX, DenseCanonicalGaussian(xPosterior.m, xPosterior.v), model.covFunc, model.covFuncParams, model.likNoiseLogStdDev)
      //      UnivariateGaussian(xTestPrior._1(0), xTestPrior._2(0, 0))

      taskId.toInt -> taskPosterior
    }.toList.toMap

    taskPosteriorByTaskId
  }

}