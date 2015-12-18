package dk.gp.hgpc

import breeze.linalg._
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.bayes.dsl.factor._
import dk.bayes.math.gaussian.canonical.CanonicalGaussian
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.gp.gp.gpPredictSingle
import dk.gp.gpc.util.calcLoglikGivenLatentVar
import dk.gp.hgpc.util.TaskVariable
import dk.gp.math.MultivariateGaussian
import dk.gp.gpc.util.createLikelihoodVariables
import dk.gp.hgpc.util.createHgpcFactorGraph
import com.typesafe.scalalogging.slf4j.LazyLogging
import dk.bayes.dsl.epnaivebayes.EPNaiveBayesFactorGraph
import dk.gp.hgpc.util.HgpcFactorGraph2
import dk.gp.hgpc.util.calibrateHgpcFactorGraph2
import dk.gp.gp.ConditionalGPFactory

/**
 * Hierarchical Gaussian Process classification. Multiple Gaussian Processes for n tasks with a single shared parent GP.
 */
object hgpcPredict2 extends LazyLogging {

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
      val predictedProb = calcLoglikGivenLatentVar(tTestPrior.m(0), tTestPrior.v(0, 0), 1d)

      predictedProb
    }.toArray
    DenseVector(predictedArray)
  }

  private def createTaskPosteriorByTaskId(xTest: DenseMatrix[Double], model: HgpcModel): Map[Int, TaskPosterior] = {

    val now = System.currentTimeMillis()
    logger.info("Calibrating factor graph...")
    val hgpcFactorGraph = HgpcFactorGraph2(model)
    val (calib, iters) = calibrateHgpcFactorGraph2(hgpcFactorGraph, maxIter = 10)
    if (iters >= 10) logger.warn(s"Factor graph did not converge in less than 10 iterations")
    logger.info("Calibrating factor graph...done: " + (System.currentTimeMillis() - now))

    val uPosterior = hgpcFactorGraph.uVariable.get.asInstanceOf[DenseCanonicalGaussian]

    val testTaskIds = xTest(::, 0).toArray.map(_.toInt).distinct
    val taskPosteriorByTaskId: Map[Int, TaskPosterior] = testTaskIds.map { taskId =>
      val idx = model.x(::, 0).findAll { x => x == taskId }
      val taskX = model.x(idx, ::).toDenseMatrix
      val taskY = model.y(idx).toDenseVector

      val taskPosterior = if (taskY.size == 0) TaskPosterior(model.u, uPosterior)
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

        TaskPosterior(taskXX, taskPosteriorXX)

      }

      taskId.toInt -> taskPosterior
    }.toList.toMap

    taskPosteriorByTaskId
  }

}