package dk.gp.gpc.util

import dk.bayes.factorgraph2.variable.CanonicalGaussianVariable
import dk.gp.gpc.GpcModel
import dk.bayes.factorgraph2.variable.BernVariable
import dk.bayes.factorgraph2.factor.StepFunctionFactor
import dk.bayes.factorgraph2.factor.CanonicalGaussianFactor
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

case class GpcFactorGraph(model: GpcModel) {

  private val covX = model.covFunc.cov(model.x, model.x, model.covFuncParams) + DenseMatrix.eye[Double](model.x.rows) * 1e-7
  private val meanX = DenseVector.zeros[Double](model.x.rows) + model.gpMean

  /**
   * Create variables
   */

  val fVariable = CanonicalGaussianVariable()

  val yVariables = model.y.toArray.map { y =>
    val k = if (y == 1) 1 else 0
    BernVariable(k)
  }

  /**
   * Create factors
   */

  val fFactor = CanonicalGaussianFactor(fVariable, meanX, covX)

  val yFactors = model.y.toArray.zipWithIndex.map {
    case (y, i) =>
      StepFunctionFactor(fVariable, yVariables(i), model.x.rows, i, v = 1)
  }

}