package dk.gp.cov

import scala.math._

/**
 * Implementation based 'http://www.gaussianprocess.org/gpml/code/matlab/doc/index.html'
 *
 *  Squared Exponential covariance function with isotropic distance measure. The
 * covariance function is parameterized as:
 *
 * k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
 *
 * where the P matrix is ell^2 times the unit matrix and sf^2 is the signal
 * variance.
 *
 * Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
 *
 * @param sf - log of signal standard deviation
 * @param ell - log of length scale standard deviation
 */

case class CovSEiso(sf: Double, ell: Double) extends CovFunc {

  def cov(x1: Array[Double], x2: Array[Double]): Double = {
    require(x1.size == x2.size, "Vectors x1 and x2 have different sizes")
    val expArg = -0.5 * distance(x1, x2, exp(2 * ell))
    exp(2 * sf) * exp(expArg)
  }

  private def distance(x1: Array[Double], x2: Array[Double], l: Double): Double = {

    var distance = 0d
    var i = 0

    while (i < x1.size) {
      distance += pow(x1(i) - x2(i), 2) / l
      i += 1
    }

    distance
  }

}