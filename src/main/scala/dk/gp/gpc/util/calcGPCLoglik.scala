package dk.gp.gpc.util

import dk.gp.gpc.GpcModel
import dk.bayes.dsl.variable.gaussian.multivariate.MultivariateGaussian
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import dk.bayes.dsl.variable.categorical.MvnGaussianThreshold
import dk.bayes.math.gaussian.canonical._
import dk.bayes.math.gaussian.canonical.CanonicalGaussian._
import dk.bayes.dsl.epnaivebayes.EPNaiveBayesFactorGraph
import breeze.linalg._
import breeze.numerics._

object calcGPCLoglik {

  def apply(model: GpcModel): Double = {

    val gpcFactorGraph = GpcFactorGraph(model)
    val (calib, iters) = calibrateGpcFactorGraph(gpcFactorGraph, maxIter = 10)

    apply(gpcFactorGraph)
  }

  def apply(calibratedFactorGraph: GpcFactorGraph): Double = {
 
    val totalLoglik = calibratedFactorGraph.yFactors.map { yFactor =>
      val outcome1Prob = yFactor.calcNewMsgV2()
    
      if (yFactor.v2.k == 1) log(outcome1Prob) else log1p(-outcome1Prob)
    }.sum

    totalLoglik
  }
  
   
  /**
   * This function will update calibratedHgpcFactorGraph for the purpose of computing approximated loglikelihood of evidence given provided covFuncParams and gpMean parameters
   * It is used for computing approximated derivatives of approximated loglikelihood(lowerBound) for covFuncParams and gpMean parameters
   */
   def apply(calibratedHgpcFactorGraph: GpcFactorGraph,covFuncParams: DenseVector[Double], gpMean: Double): Double = {
     
     calibratedHgpcFactorGraph.updateFfactor(covFuncParams, gpMean)
     calibratedHgpcFactorGraph.fFactor.updateMsgV1()
     calibratedHgpcFactorGraph.fVariable.update()
     
     apply(calibratedHgpcFactorGraph)
   }
}