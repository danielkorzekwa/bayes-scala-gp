package dk.gp.hgpc.util

import dk.bayes.factorgraph2.variable.BernVariable
import dk.bayes.factorgraph2.variable.CanonicalGaussianVariable
import dk.gp.gp.ConditionalGPFactory
import dk.bayes.factorgraph2.factor.CanonicalLinearGaussianFactor
import dk.bayes.factorgraph2.factor.CanonicalGaussianFactor
import dk.gp.hgpc.HgpcModel
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.bayes.factorgraph2.factor.StepFunctionFactor
import dk.bayes.factorgraph2.variable.BernVariable
import dk.bayes.factorgraph2.factor.StepFunctionFactor

case class HgpcFactorGraph2(model: HgpcModel) {

  private val x = model.x
  private val y = model.y
  private val u = model.u

  private val covFunc = model.covFunc
  private val covFuncParams = model.covFuncParams
  private val mean = model.mean

  private val covU = covFunc.cov(u, u, covFuncParams) + DenseMatrix.eye[Double](u.rows) * 1e-7
  private val meanU = DenseVector.zeros[Double](u.rows) + mean

  val taskIds = x(::, 0).toArray.map(_.toInt).distinct

  /**
   * Create variables
   */

  val uVariable = CanonicalGaussianVariable()

  val taskVariablesMap = taskIds.map(taskId => taskId -> CanonicalGaussianVariable()).toMap

  val taskYVariablesMap: Map[Int, Array[BernVariable]] = taskIds.map { taskId =>

    val idx = x(::, 0).findAll { x => x == taskId }
    val taskY = y(idx).toDenseVector

    val yVariables = taskY.toArray.map { yPoint =>
      val k = if (yPoint == 1) 1 else 0

      BernVariable(k)
    }

    taskId -> yVariables
  }.toMap

  /**
   * Create factors
   */
  val uFactor = CanonicalGaussianFactor(uVariable, meanU, covU)

  val taskFactorsMap = taskIds.map { taskId =>
    val idx = x(::, 0).findAll { x => x == taskId }
    val taskX = x(idx, ::).toDenseMatrix
    val (a, b, v) = ConditionalGPFactory(u, covFunc, covFuncParams, mean).create(taskX)
    taskId -> CanonicalLinearGaussianFactor(uVariable, taskVariablesMap(taskId), a, b, v)
  }.toMap

  val taskYFactorsMap: Map[Int, Array[StepFunctionFactor]] = taskIds.map { taskId =>

    val idx = x(::, 0).findAll { x => x == taskId }
    val taskY = y(idx).toDenseVector

    val yFactors = taskY.toArray.zipWithIndex.map {
      case (y, i) => StepFunctionFactor(taskVariablesMap(taskId), taskYVariablesMap(taskId)(i), taskY.size, i, v = 1)

    }

    taskId -> yFactors
  }.toMap

}