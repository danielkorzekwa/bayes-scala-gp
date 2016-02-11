package dk.gp.gpc.util

import breeze.linalg.DenseVector
import dk.bayes.math.gaussian.canonical.CanonicalGaussian
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.bayes.math.gaussian.canonical.SparseCanonicalGaussian
import breeze.linalg.DenseMatrix
import dk.bayes.math.gaussian.Gaussian
import breeze.numerics._
import breeze.linalg._
import breeze.stats._

/**
 * Following Daniel Korzekwa. Predictive log likelihood for Gaussian Process Classification with noise step function likelihood, 01 Jan 2016
 */
object calcGPCLoglikMeanD {

  def apply(factorGraph: GpcFactorGraph): Double = {

    val fPrior = factorGraph.fFactor.getMsgV1().get.asInstanceOf[DenseCanonicalGaussian]
    val fPosterior = factorGraph.fVariable.get.asInstanceOf[DenseCanonicalGaussian]

    val dFPriorMu = DenseVector.fill(fPrior.mean.size)(1d)
    val dFPostH = fPrior.k * dFPriorMu //eq.20 -> eq.24 
    val dFPostMu = fPosterior.variance * dFPostH //eq.16  

    val loglikD_i = factorGraph.yFactors.zipWithIndex.map { case (yFactor, i) => eq4_i(factorGraph, dFPostMu, i) }

    loglikD_i.sum

  }

  def eq4_i(factorGraph: GpcFactorGraph, dFPostMu: DenseVector[Double], i: Int): Double = {

    val yFactor = factorGraph.yFactors(i)

    val y = if (yFactor.v2.k == 1) 1d else -1d
    val cavity = yFactor.calcCavity()
    val mu = cavity.m
    val s2 = cavity.v + 1

    val dMu = eq_8(cavity, factorGraph.fVariable.get.asInstanceOf[DenseCanonicalGaussian], dFPostMu, i)
    val t1 = 1d / Gaussian.stdCdf(y * (mu / sqrt(s2)))
    val t2 = Gaussian.stdPdf(y * (mu / sqrt(s2)))
    val t3 = y * (dMu / sqrt(s2))

    val t = t1 * t2 * t3

    t

  }

  def eq_8(cavity: DenseCanonicalGaussian, fPosterior: DenseCanonicalGaussian, dFPostMu: DenseVector[Double], i: Int): Double = {

    val dCavH = (1d / fPosterior.variance(i, i)) * dFPostMu(i) //eq.12
    val dCavMu = cavity.v * dCavH //eq.8
    dCavMu
  }

}