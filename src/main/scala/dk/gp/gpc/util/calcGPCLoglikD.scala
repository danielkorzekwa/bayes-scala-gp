package dk.gp.gpc.util

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.numerics.sqrt
import dk.bayes.math.gaussian.Gaussian
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian.toGaussian

/**
 * Following Daniel Korzekwa. Predictive log likelihood for Gaussian Process Classification with noise step function likelihood, 01 Jan 2016
 */
object calcGPCLoglikD {

  def apply(factorGraph: GpcFactorGraph): DenseVector[Double] = {

    val dFPriorS2 = factorGraph.model.covFunc.covD(factorGraph.model.x, factorGraph.model.x, factorGraph.model.covFuncParams)
    val fPrior = factorGraph.fFactor.getMsgV1().get.asInstanceOf[DenseCanonicalGaussian]
    val fPosterior = factorGraph.fVariable.get.asInstanceOf[DenseCanonicalGaussian]

    val loglikDArray = dFPriorS2.map { dFPriorS2 =>

      val dPostK = -fPrior.k * dFPriorS2 * fPrior.k //eq.19 -> eq.23
      val dFPostS2 = -fPosterior.variance * dPostK * fPosterior.variance //eq. 15
      val dPostH = dPostK * fPrior.mean //eq.20 -> eq.24) //dFPriorMu=0
      val dFPostMu = dFPostS2 * fPosterior.h + fPosterior.variance * dPostH //eq.16  

      val loglikD_i = factorGraph.yFactors.zipWithIndex.map { case (yFactor, i) => eq4_i(factorGraph, dFPostS2, dFPostMu, i) }

      loglikD_i.sum
    }

    DenseVector(loglikDArray)
  }

  def eq4_i(factorGraph: GpcFactorGraph, dFPostS2: DenseMatrix[Double], dFPostMu: DenseVector[Double], i: Int): Double = {

    val yFactor = factorGraph.yFactors(i)

    val y = if (yFactor.v2.k == 1) 1d else -1d
    val cavity = yFactor.calcCavity()
    val mu = cavity.m
    val s2 = cavity.v + 1

    val (dS2, dMu) = eq7_8_i(factorGraph, cavity, factorGraph.fVariable.get.asInstanceOf[DenseCanonicalGaussian], i, dFPostS2, dFPostMu)

    val t1 = 1d / Gaussian.stdCdf(y * (mu / sqrt(s2)))
    val t2 = Gaussian.stdPdf(y * (mu / sqrt(s2)))
    val t3 = y * (dMu / sqrt(s2) - (mu * dS2) / (2d * s2 * sqrt(s2)))

    val t = t1 * t2 * t3

    t

  }

  def eq7_8_i(factorGraph: GpcFactorGraph, cavity: DenseCanonicalGaussian, fPosterior: DenseCanonicalGaussian, i: Int, dFPostS2: DenseMatrix[Double], dFPostMu: DenseVector[Double]): (Double, Double) = {

    val cavS2 = cavity.v

    val fPrior = factorGraph.fFactor.getMsgV1().get.asInstanceOf[DenseCanonicalGaussian]

    val dCavK = -(1d / fPosterior.variance(i, i)) * dFPostS2(i, i) * (1d / fPosterior.variance(i, i)) //eq.11
    val dCavS2 = -cavS2 * dCavK * cavS2 //eq. 7

    val dCavH = dCavK * fPosterior.mean(i) + (1d / fPosterior.variance(i, i)) * dFPostMu(i) //eq.12
    val dCavMu = dCavS2 * cavity.h(0) + cavity.v * dCavH //eq.8

    (dCavS2, dCavMu)
  }

}