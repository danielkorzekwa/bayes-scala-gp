package dk.gp.sgpr.lb

import breeze.linalg.sum
import breeze.numerics.sqrt
import dk.gp.math.diagProd

/**
 * Calculates derivatives of variational lower bound with respect to covariance hyimport dk.gp.sgpr.lb.SgprLowerBound
per parameters for Sparse GP regression model.
 *
 * Based on:
 * - Variational Learning of Inducing Variables in Sparse Gaussian Processes (http://jmlr.org/proceedings/papers/v5/titsias09a/titsias09a.pdf)
 * - Derivatives of lower bound, Michalis K. Titsias
 */
object calcLowerBoundDKernel {

  def apply(lb: SgprLowerBound): Array[Double] = {

    val expensiveTerm = lb.kMMinv / lb.likNoiseVar - lb.aInv - (lb.aInvkMNy / sqrt(lb.likNoiseVar)) * (lb.aInvkMNy.t / sqrt(lb.likNoiseVar))
    val t1247_t2 = (lb.kMMinv / lb.likNoiseVar) * lb.kMNkNM * (lb.kMMinv / lb.likNoiseVar)
    val t3568_t1 = expensiveTerm * lb.kMN + (lb.aInvkMNy / lb.likNoiseVar) * lb.y.t // [m x n]

    val dArray = (0 until lb.kMMdArray.size).map { i =>
      val kMMd = lb.kMMdArray(i)
      val kNMd = lb.kNMdArray(i)
      val kNNDiagD = lb.kNNDiagDArray(i)

      val t1247 = 0.5 * lb.likNoiseVar * sum(diagProd(kMMd, (expensiveTerm - t1247_t2).t))
      val t3568 = sum(diagProd(t3568_t1, kNMd.t))

      val f4 = -0.5 / lb.likNoiseVar * sum(kNNDiagD)
      t1247 + t3568 + f4
    }.toArray

    dArray
  }

}