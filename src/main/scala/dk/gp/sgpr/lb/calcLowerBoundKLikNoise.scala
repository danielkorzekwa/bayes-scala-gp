package dk.gp.sgpr.lb

import breeze.linalg.sum

/**
 * Calculates derivative of variational lower bound with respect to likelihood log noise std dev for Sparseimport dk.gp.sgpr.lb.SgprLowerBound
 GP regression model.
 *
 * Based on:
 * - Variational Learning of Inducing Variables in Sparse Gaussian Processes (http://jmlr.org/proceedings/papers/v5/titsias09a/titsias09a.pdf)
 * - Derivatives of lower bound, Michalis K. Titsias
 */
object calcLowerBoundKLikNoise {

  /**
   * @param lowerBoundTerms 6 terms of lower bound
   *
   */
  def apply(lb: SgprLowerBound, lbTerm3: Double, lbTerm4: Double, lbTerm5: Double): Double = {

    val n = lb.x.rows
    val m = lb.u.rows

    val f0Term = -(n - m) + ((1d / lb.likNoiseVar) * lb.yy)

    val yKnmInvLmInvLainvLa =lb.yKnmInvLmInvLa * lb.invLa.t
    val aux = lb.likNoiseVar * sum(lb.invLa :* lb.invLa) + yKnmInvLmInvLainvLa * yKnmInvLmInvLainvLa.t

    val f23 = -2 * lbTerm3 - aux
    f0Term + f23 - 2 * (lbTerm4 + lbTerm5)
  }

}