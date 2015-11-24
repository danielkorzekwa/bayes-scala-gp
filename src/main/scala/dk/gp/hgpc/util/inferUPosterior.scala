package dk.gp.hgpc.util

import dk.gp.hgpc.HgpcModel
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import breeze.linalg.DenseMatrix
import dk.bayes.infer.epnaivebayes.EPNaiveBayesFactorGraph
import breeze.linalg.DenseVector
import dk.bayes.dsl.factor.DoubleFactor
import dk.bayes.math.gaussian.canonical.CanonicalGaussian

object inferUPosterior {

  def apply(model: HgpcModel): DenseCanonicalGaussian = {
    val covU = model.covFunc.cov(model.u, model.u, model.covFuncParams) + DenseMatrix.eye[Double](model.u.rows) * 1e-7
    val meanU = DenseVector.zeros[Double](model.u.rows) + model.mean

    val uVariable = dk.bayes.dsl.variable.gaussian.multivariate.MultivariateGaussian(meanU, covU)

    val taskIds = model.x(::, 0).toArray.distinct
    val taskVariables = taskIds.map { taskId =>

      val idx = model.x(::, 0).findAll { x => x == taskId }
      val taskX = model.x(idx, ::).toDenseMatrix
      val taskY = model.y(idx).toDenseVector

      TaskVariable(taskX, taskY, model, uVariable).asInstanceOf[DoubleFactor[CanonicalGaussian, _]] //@TODO this is a hack to allow for using a proper implicit for multOp()
    }

    val factorGraph = EPNaiveBayesFactorGraph(uVariable, taskVariables, true)
    factorGraph.calibrate(maxIter = 10, threshold = 1e-4)
    val uPosterior = factorGraph.getPosterior().asInstanceOf[DenseCanonicalGaussian]

    uPosterior
  }
}