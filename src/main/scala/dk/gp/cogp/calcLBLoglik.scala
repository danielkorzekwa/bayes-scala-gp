package dk.gp.cogp

import dk.gp.math.MultivariateGaussian
import breeze.numerics._
import breeze.linalg.DenseMatrix
import breeze.linalg.logdet
import breeze.linalg.inv
import breeze.linalg.trace
import dk.gp.math.MultivariateGaussian
import breeze.linalg.DenseVector
import breeze.linalg.sum
import scala.math.Pi
import breeze.linalg.diag
import dk.gp.cov.utils.covDiag

object calcLBLoglik {

  def apply(cogpModel: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): Double = {

    val z = x
    val kXZ = cogpModel.g.head.covFunc.cov(z, z, cogpModel.g.head.covFuncParams)
    val kZZ = cogpModel.g.head.covFunc.cov(z, z, cogpModel.g.head.covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)
    val kXXDiag = covDiag(x, cogpModel.g.head.covFunc, cogpModel.g.head.covFuncParams)

    val gArray = cogpModel.g
    val hArray = cogpModel.h
    val beta = cogpModel.beta
    val w = cogpModel.w

    val Aj = kXZ * inv(kZZ)
    val Ai = Aj
    val lambdaJ = Aj.t * Aj
    val lambdaI = lambdaJ

    val kZX = kXZ.t
    val kTildeDiagSum = sum(kXXDiag - diag(kXZ * inv(kZZ) * kZX))

    val qTerm = (0 until gArray.size).map { j =>

      val u = gArray(j).u
      val qTerm_j = 0.5 * (logdet(kZZ)._2 + logdet(inv(u.v))._2) + 0.5 * trace(inv(kZZ) * (u.m * u.m.t + u.v))
      qTerm_j
    }.sum

    val pTerm = (0 until hArray.size).map { i =>

      val v = hArray(i).u
      val pTerm_i = 0.5 * (logdet(kZZ)._2 + logdet(inv(v.v))._2) + 0.5 * trace(inv(kZZ) * ((v.m * v.m.t + v.v)))

      val traceQTerm = (0 until gArray.size).map { j =>
        val u = gArray(j).u
        pow(w(i, j), 2) * trace(u.v * lambdaJ)
      }.sum

      val kTildeQTerm = (0 until gArray.size).map { j =>
        pow(w(i, j), 2) * kTildeDiagSum
      }.sum

      val tracePTerm = beta(i) * 0.5 * trace(v.v * lambdaI)
      val kTildePTerm = 0.5 * beta(i) * kTildeDiagSum

      val wAm = (0 until gArray.size).foldLeft(DenseVector.zeros[Double](kXZ.rows))((wAm, j) => wAm + w(i, j) * Aj * gArray(j).u.m)
      val yTerm = y(::, i) - wAm - Ai * hArray(i).u.m

      val logNTerm = -0.5 * y.size * log(2 * Pi / beta(i)) - 0.5 * beta(i) * sum(pow(yTerm, 2))

      pTerm_i + logNTerm - 0.5 * beta(i) * traceQTerm - 0.5 * beta(i) * kTildeQTerm - tracePTerm - kTildePTerm

    }.sum

    -qTerm - pTerm

  }

}