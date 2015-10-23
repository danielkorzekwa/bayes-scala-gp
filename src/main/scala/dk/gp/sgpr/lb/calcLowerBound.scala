package dk.gp.sgpr.lb

import breeze.linalg.DenseVector
import breeze.linalg.cholesky
import breeze.linalg.sum
import breeze.numerics._
import dk.gp.math.diagProd
import dk.gp.math.logdetchol
import scala.math.Pi

/** Variational lower bound for Sparse GP regression model.
 *
 * Based on:
 * - Variational Learning of Inducing Variables in Sparse Gaussian Processes (http://jmlr.org/proceedings/papers/v5/titsias09a/titsias09a.pdf)
 * - Derivatives of lower bound, Michalis K. Titsias
 */
object calcLowerBound {

  def apply(lb: SgprLowerBound): Array[Double] = {

    val n = lb.x.rows
    val m = lb.u.rows
    
    val f0Term =  (-n / 2) * log(2 * Pi) - ((n - m).toDouble / 2) * log(lb.likNoiseVar) - (1d / (2 * lb.likNoiseVar)) * lb.yy

    val f12 = -0.5 * logdetchol(lb.la)

    val f3Term = (1d / (2 * lb.likNoiseVar)) * (lb.yKnmInvLmInvLa * lb.yKnmInvLmInvLa.t)

    val f4Term = -(1d / (2 * lb.likNoiseVar)) * sum(lb.kNNdiag)
    val f5Term = (1d / (2 * lb.likNoiseVar)) * sum(diagProd(lb.kMMinv, lb.kMNkNM.t))

    Array(f0Term, f12, f3Term, f4Term, f5Term)
  }
   

}