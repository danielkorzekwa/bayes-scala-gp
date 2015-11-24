package dk.gp.gpc.util

import dk.gp.math.MultivariateGaussian
import breeze.linalg.DenseVector
import DenseVector._
import breeze.linalg._
import dk.bayes.math.gaussian.Gaussian
import breeze.numerics._

object calcLoglikGivenLatentVar {

  def apply(xMean: DenseVector[Double], xVar: DenseMatrix[Double], stepFunctionNoiseVar: Double): DenseVector[Double] = {

    val meanVar = horzcat(xMean, diag(xVar))
    val yProbs = meanVar(*, ::).map(x => apply(x(0),x(1),stepFunctionNoiseVar))
    yProbs
  }
  
  def apply(xMean:Double,xVariance:Double,stepFunctionNoiseVar: Double):Double =  Gaussian.stdCdf(xMean / sqrt(stepFunctionNoiseVar + xVariance)) 
}