package dk.gp.gpc.util

import dk.gp.gpc.GpcModel
import dk.bayes.infer.epnaivebayes.EPNaiveBayesFactorGraph
import dk.bayes.dsl.variable.gaussian.multivariate.MultivariateGaussian
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import dk.bayes.dsl.variable.categorical.MvnGaussianThreshold
import dk.bayes.math.gaussian.canonical._
import dk.bayes.math.gaussian.canonical.CanonicalGaussian._
object calcGPCLoglik {

  def apply(model: GpcModel): Double = {

    val covX = model.covFunc.cov(model.x, model.x, model.covFuncParams) + DenseMatrix.eye[Double](model.x.rows) * 1e-7
    val meanX = DenseVector.zeros[Double](model.x.rows) + model.mean

    val fVariable = MultivariateGaussian(meanX, covX)
    val yVariables = createLikelihoodVariables(fVariable,model.y)
    val factorGraph = EPNaiveBayesFactorGraph(fVariable, yVariables, true)
    factorGraph.calibrate(maxIter = 10, threshold = 1e-4)

    val fPosterior = factorGraph.getPosterior().asInstanceOf[DenseCanonicalGaussian]
    val logliks = factorGraph.getMsgsUp().zip(yVariables).map { case (msgUp, yVariable) => yVariable.loglik(fPosterior, msgUp.asInstanceOf[SparseCanonicalGaussian]) }

    logliks.sum
  }
}