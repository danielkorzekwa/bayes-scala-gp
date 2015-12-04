package dk.gp.hgpc.util

import dk.gp.hgpc.HgpcModel
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.bayes.dsl.epnaivebayes.EPNaiveBayesFactorGraph

object createHgpcFactorGraph {
  
  def apply(model: HgpcModel):EPNaiveBayesFactorGraph[DenseCanonicalGaussian] = {
    
      val covU = model.covFunc.cov(model.u, model.u, model.covFuncParams) + DenseMatrix.eye[Double](model.u.rows) * 1e-7
    val meanU = DenseVector.zeros[Double](model.u.rows) + model.mean

    val uVariable = dk.bayes.dsl.variable.gaussian.multivariate.MultivariateGaussian(meanU, covU)

    val taskIds = model.x(::, 0).toArray.distinct
    val taskVariables = taskIds.map { taskId =>

      val idx = model.x(::, 0).findAll { x => x == taskId }
      val taskX = model.x(idx, ::).toDenseMatrix
      val taskY = model.y(idx).toDenseVector

      TaskVariable(taskX, taskY, model, uVariable)
    }

    val factorGraph = EPNaiveBayesFactorGraph(uVariable, taskVariables, true)
    factorGraph
  }
}