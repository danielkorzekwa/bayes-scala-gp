package dk.gp.hgpc.util

import dk.gp.hgpc.HgpcModel
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.bayes.infer.epnaivebayes.EPNaiveBayesFactorGraph
import dk.bayes.dsl.factor.DoubleFactor
import dk.bayes.math.gaussian.canonical.CanonicalGaussian
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.bayes.math.gaussian.canonical.SparseCanonicalGaussian

object calcHGPCLoglik {

  def apply(model: HgpcModel): Double = {

    val hgpcFactorGraph = createHgpcFactorGraph(model)
    hgpcFactorGraph.calibrate(maxIter = 10, threshold = 1e-4)
    val uPosterior = hgpcFactorGraph.getPosterior().asInstanceOf[DenseCanonicalGaussian]

    val logliks = hgpcFactorGraph.getMsgsUp().zip(hgpcFactorGraph.likelihoods).map { case (msgUp, taskVariable) => taskVariable.asInstanceOf[TaskFactor].calcLoglik(uPosterior, msgUp.asInstanceOf[DenseCanonicalGaussian]) }

    logliks.sum
  }
}