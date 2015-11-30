package dk.gp.hgpc.util

import dk.bayes.dsl.factor.DoubleFactor
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import dk.gp.gp.gpPredict
import dk.gp.gp.gpPredictSingle
import dk.bayes.dsl.variable.gaussian.multivariate.MultivariateGaussian
import dk.gp.gpc.util.createLikelihoodVariables
import dk.bayes.infer.epnaivebayes.EPNaiveBayesFactorGraph
import dk.gp.gp.ConditionalGPFactory
import dk.gp.gpc.util.calcLoglikGivenLatentVar
import dk.bayes.math.gaussian.canonical.SparseCanonicalGaussian
import breeze.linalg.MatrixNotSymmetricException

trait TaskFactor extends DoubleFactor[DenseCanonicalGaussian, Any] {

  this: TaskVariable =>

  val initFactorMsgUp: DenseCanonicalGaussian = {
    val m = DenseVector.zeros[Double](this.uVariable.m.size)
    val v = DenseMatrix.eye[Double](this.uVariable.m.size) * 1000d
    DenseCanonicalGaussian(m, v)
  }

  def calcLoglik(uPosterior: DenseCanonicalGaussian, oldFactorMsgUp: DenseCanonicalGaussian): Double = {
    val uVarMsgDown = uPosterior / oldFactorMsgUp
    val taskXPrior = gpPredictSingle(this.taskX, dk.gp.math.MultivariateGaussian(uVarMsgDown.mean, uVarMsgDown.variance), this.model.u, this.model.covFunc, this.model.covFuncParams, this.model.mean)

    val taskXPriorVariable = MultivariateGaussian(taskXPrior.m, taskXPrior.v)

    val yVariables = createLikelihoodVariables(taskXPriorVariable, this.taskY)

    val factorGraph = EPNaiveBayesFactorGraph(taskXPriorVariable, yVariables, true)
    factorGraph.calibrate(maxIter = 10, threshold = 1e-4)

    val taskXPosteriorVariable = factorGraph.getPosterior().asInstanceOf[DenseCanonicalGaussian]

    val logliks = factorGraph.getMsgsUp().zip(yVariables).map { case (msgUp, yVariable) => yVariable.loglik(taskXPosteriorVariable, msgUp.asInstanceOf[SparseCanonicalGaussian]) }

    logliks.sum
  }

  def calcYFactorMsgUp(uPosterior: DenseCanonicalGaussian, oldFactorMsgUp: DenseCanonicalGaussian): Option[DenseCanonicalGaussian] = {

    val uVarMsgDown = uPosterior / oldFactorMsgUp

    val taskXPrior = gpPredictSingle(this.taskX, dk.gp.math.MultivariateGaussian(uVarMsgDown.mean, uVarMsgDown.variance), this.model.u, this.model.covFunc, this.model.covFuncParams, this.model.mean)

    val taskXPriorVariable = MultivariateGaussian(taskXPrior.m, taskXPrior.v)

    val yVariables = createLikelihoodVariables(taskXPriorVariable, this.taskY)

    val factorGraph = EPNaiveBayesFactorGraph(taskXPriorVariable, yVariables, true)
   factorGraph.calibrate(maxIter = 10, threshold = 1e-4)
    
    val taskXPosteriorVariable = factorGraph.getPosterior().asInstanceOf[DenseCanonicalGaussian]

    val taskXVarMsgUp = taskXPosteriorVariable / DenseCanonicalGaussian(taskXPriorVariable.m, taskXPriorVariable.v)

    val (a, b, v) = ConditionalGPFactory(this.model.u, this.model.covFunc, this.model.covFuncParams, this.model.mean).create(this.taskX)
    val xFactorCanon = DenseCanonicalGaussian(a, b, v)

    val factorTimesMsg = xFactorCanon * taskXVarMsgUp.extend(a.cols + a.rows, a.cols)
    val newXFactorMsgUp = factorTimesMsg.marginal((0 until a.cols): _*)
    Some(newXFactorMsgUp)
  }
}