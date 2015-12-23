package dk.gp.hgpc

import com.typesafe.scalalogging.slf4j.LazyLogging

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.gp.gp.ConditionalGPFactory
import dk.gp.gp.GPPredictSingle
import dk.gp.gpc.util.calcLoglikGivenLatentVar
import dk.gp.hgpc.util.HgpcFactorGraph
import dk.gp.hgpc.util.calibrateHgpcFactorGraph
import dk.gp.math.MultivariateGaussian

/**
 * Hierarchical Gaussian Process classification. Multiple Gaussian Processes for n tasks with a single shared parent GP.
 */
object hgpcPredict extends LazyLogging {

  case class TaskPosterior(x: DenseMatrix[Double], xPosterior: DenseCanonicalGaussian)

  /**
   * Returns vector of probabilities for a class 1
   */
  def apply(t: DenseMatrix[Double], model: HgpcModel): DenseVector[Double] = {
    val gpModelsByTaskId: Map[Int, GPPredictSingle] = createTaskPosteriorByTaskId(t, model)

    val predictedArray = (0 until t.rows).par.map { rowIndex =>

      val tRow = t(rowIndex, ::).t
      val taskId = tRow(0).toInt
      val gpModel = gpModelsByTaskId(taskId)

      val tTestPrior = gpModel.predictSingle(tRow.toDenseMatrix)
      val predictedProb = calcLoglikGivenLatentVar(tTestPrior.m(0), tTestPrior.v(0, 0), 1d)

      predictedProb
    }.toArray
    DenseVector(predictedArray)
  }

  private def createTaskPosteriorByTaskId(xTest: DenseMatrix[Double], model: HgpcModel): Map[Int, GPPredictSingle] = {

    val now = System.currentTimeMillis()
    logger.info("Calibrating factor graph...")
    val hgpcFactorGraph = HgpcFactorGraph(model)
    val (calib, iters) = calibrateHgpcFactorGraph(hgpcFactorGraph, maxIter = 10)
    if (iters >= 10) logger.warn(s"Factor graph did not converge in less than 10 iterations")
    logger.info("Calibrating factor graph...done: " + (System.currentTimeMillis() - now))

    val uPosterior = hgpcFactorGraph.uVariable.get.asInstanceOf[DenseCanonicalGaussian]

    logger.info("Computing taskPosteriorByTaskId...")

    val testTaskIds = xTest(::, 0).toArray.map(_.toInt).distinct
    val gpModelsByTaskId: Map[Int, GPPredictSingle] = testTaskIds.map { taskId =>
      val idx = model.x(::, 0).findAll { x => x == taskId }
      val taskX = model.x(idx, ::).toDenseMatrix
      val taskY = model.y(idx).toDenseVector

      val gpModel = if (taskY.size == 0) GPPredictSingle(MultivariateGaussian(uPosterior.mean, uPosterior.variance), model.u, model.covFunc, model.covFuncParams, model.mean)
      else {

        val taskXTestIdx = xTest(::, 0).findAll(x => x == taskId)
        val taskXTest = xTest(taskXTestIdx, ::).toDenseMatrix

        val taskXX = DenseMatrix.vertcat(taskX, taskXTest)

        val condGPFactory = ConditionalGPFactory(model.u, model.covFunc, model.covFuncParams)
        val (a, b, v) = condGPFactory.create(taskXX)
        val taskFactorDownMsgFull = hgpcFactorGraph.taskFactorsMap(taskId).calcNewMsgV2(a, b, v).asInstanceOf[DenseCanonicalGaussian]

        val taskVarPosterior = hgpcFactorGraph.taskVariablesMap(taskId).get.asInstanceOf[DenseCanonicalGaussian]

        val taskFactorDownMsg = hgpcFactorGraph.taskFactorsMap(taskId).getMsgV2().get.asInstanceOf[DenseCanonicalGaussian]
        val taskPosteriorXX = taskFactorDownMsgFull * ((taskVarPosterior / taskFactorDownMsg).extend(taskXX.rows, 0))

        GPPredictSingle(MultivariateGaussian(taskPosteriorXX.mean, taskPosteriorXX.variance), taskXX, model.covFunc, model.covFuncParams, model.mean)

      }

      taskId.toInt -> gpModel
    }.toList.toMap

    logger.info("Computing taskPosteriorByTaskId...done")

    gpModelsByTaskId
  }

}