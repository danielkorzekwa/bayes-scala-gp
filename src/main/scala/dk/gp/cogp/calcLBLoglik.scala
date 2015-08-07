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

    val gArray = cogpModel.g
    val hArray = cogpModel.h
    val beta = cogpModel.beta
    val w = cogpModel.w

    val qTerm = (0 until gArray.size).map { j =>

      val z = cogpModel.g(j).z
      val kZZ = cogpModel.g(j).covFunc.cov(z, z, cogpModel.g(j).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)

      val u = gArray(j).u
      val qTerm_j = 0.5 * (logdet(kZZ)._2 + logdet(inv(u.v))._2) + 0.5 * trace(inv(kZZ) * (u.m * u.m.t + u.v))
      qTerm_j
    }.sum

    val pTerm = (0 until hArray.size).map { i =>

      val z = cogpModel.h(i).z
      val kZZ2 = cogpModel.h(i).covFunc.cov(z, z, cogpModel.h(i).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)
      val kXZ2 = cogpModel.h(i).covFunc.cov(z, z, cogpModel.h(i).covFuncParams)
      val kZX2 = kXZ2.t
      val Ai2 = kXZ2 * inv(kZZ2)
      val lambdaI = Ai2.t * Ai2

      val kXXDiag_i = covDiag(x, cogpModel.h(i).covFunc, cogpModel.h(i).covFuncParams)

      val kTildeDiagSum_i = sum(kXXDiag_i - diag(kXZ2 * inv(kZZ2) * kZX2))

      val v = hArray(i).u
      val pTerm_i = 0.5 * (logdet(kZZ2)._2 - logdet(v.v)._2) + 0.5 * trace(inv(kZZ2) * ((v.m * v.m.t + v.v)))

      val traceQTerm = (0 until gArray.size).map { j =>

        val z = cogpModel.g(j).z
        val kXZ = cogpModel.g(j).covFunc.cov(z, z, cogpModel.g(j).covFuncParams)
        val kZZ = cogpModel.g(j).covFunc.cov(z, z, cogpModel.g(j).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)

        val Aj = kXZ * inv(kZZ)
        val lambdaJ = Aj.t * Aj
        val u = gArray(j).u
        pow(w(i, j), 2) * trace(u.v * lambdaJ)
      }.sum

      val kTildeQTerm = (0 until gArray.size).map { j =>

        val z = cogpModel.g(j).z
        val kXZ = cogpModel.g(j).covFunc.cov(z, z, cogpModel.g(j).covFuncParams)
        val kZZ = cogpModel.g(j).covFunc.cov(z, z, cogpModel.g(j).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)
        val kZX = kXZ.t
        val kXXDiag = covDiag(x, cogpModel.g(j).covFunc, cogpModel.g(j).covFuncParams)

        val kTildeDiagSum = sum(kXXDiag - diag(kXZ * inv(kZZ) * kZX))

        pow(w(i, j), 2) * kTildeDiagSum
      }.sum

      val tracePTerm = beta(i) * 0.5 * trace(v.v * lambdaI)
      val kTildePTerm = 0.5 * beta(i) * kTildeDiagSum_i

      val wAm = (0 until gArray.size).foldLeft(DenseVector.zeros[Double](x.rows)) { (wAm, j) =>

        val z = cogpModel.g(j).z
        val kXZ = cogpModel.g(j).covFunc.cov(z, z, cogpModel.g(j).covFuncParams)
        val kZZ = cogpModel.g(j).covFunc.cov(z, z, cogpModel.g(j).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)
        val Aj = kXZ * inv(kZZ)

        wAm + w(i, j) * Aj * gArray(j).u.m
      }

      val yTerm = y(::, i) - wAm - Ai2 * hArray(i).u.m

      val logNTerm = -0.5 * y.rows * log(2 * Pi / beta(i)) - 0.5 * beta(i) * sum(pow(yTerm, 2))

      pTerm_i + logNTerm - 0.5 * beta(i) * traceQTerm - 0.5 * beta(i) * kTildeQTerm - tracePTerm - kTildePTerm

      logNTerm - pTerm_i - tracePTerm - kTildePTerm - 0.5 * beta(i) * traceQTerm - 0.5 * beta(i) * kTildeQTerm

    }.sum

    -qTerm + pTerm
  }

}