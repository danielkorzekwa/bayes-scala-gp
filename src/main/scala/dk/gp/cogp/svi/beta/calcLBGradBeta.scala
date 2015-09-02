package dk.gp.cogp.svi.beta

import breeze.linalg.DenseVector
import dk.gp.cogp.CogpModel
import breeze.linalg.DenseMatrix
import breeze.linalg.inv
import breeze.numerics._
import breeze.linalg._
import dk.gp.cov.utils.covDiag
import dk.gp.math.invchol

object calcLBGradBeta {

  def apply(model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseVector[Double] = {

    val dBeta = DenseVector.zeros[Double](model.beta.size)

    val gArray = model.g
    val hArray = model.h

    val w = model.w

    for (i <- 0 until model.beta.size) {

      val z = model.h(i).z
      val kZZ2 = model.h(i).covFunc.cov(z, z, model.h(i).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)
      val kXZ2 = model.h(i).covFunc.cov(z, z, model.h(i).covFuncParams)
      val kZX2 = kXZ2.t

      val kZZ2inv = invchol(cholesky(kZZ2).t)

      val Ai2 = kXZ2 * kZZ2inv
      val lambdaI = Ai2.t * Ai2

      val kXXDiag_i = covDiag(x, model.h(i).covFunc, model.h(i).covFuncParams)
      val kTildeDiagSum_i = sum(kXXDiag_i - diag(kXZ2 * kZZ2inv * kZX2))

      val v = hArray(i).u

      val beta = model.beta(i)

      val wAm = (0 until gArray.size).foldLeft(DenseVector.zeros[Double](x.rows)) { (wAm, j) =>

        val z = model.g(j).z
        val kXZ = model.g(j).covFunc.cov(z, z, model.g(j).covFuncParams)
        val kZZ = model.g(j).covFunc.cov(z, z, model.g(j).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)

        val kZZinv = invchol(cholesky(kZZ).t)
        val Aj = kXZ * kZZinv

        wAm + w(i, j) * Aj * gArray(j).u.m
      }

      val yTerm = y(::, i) - wAm - Ai2 * hArray(i).u.m

      val logTermD = (0.5 * y.rows) / beta - 0.5 * sum(pow(yTerm, 2))

      val kTildeQTerm = (0 until gArray.size).map { j =>

        val z = model.g(j).z
        val kXZ = model.g(j).covFunc.cov(z, z, model.g(j).covFuncParams)
        val kZZ = model.g(j).covFunc.cov(z, z, model.g(j).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)
        val kZX = kXZ.t
        val kXXDiag = covDiag(x, model.g(j).covFunc, model.g(j).covFuncParams)

        val kZZinv = invchol(cholesky(kZZ).t)
        val kTildeDiagSum = sum(kXXDiag - diag(kXZ * kZZinv * kZX))

        pow(w(i, j), 2) * kTildeDiagSum
      }.sum

      val tildeQD = 0.5 * kTildeQTerm

      val traceQTerm = (0 until gArray.size).map { j =>

        val z = model.g(j).z
        val kXZ = model.g(j).covFunc.cov(z, z, model.g(j).covFuncParams)
        val kZZ = model.g(j).covFunc.cov(z, z, model.g(j).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)

        val R = cholesky(kZZ).t
        val kZZinv = invchol(R)
        val Aj = kXZ * kZZinv
        val lambdaJ = Aj.t * Aj
        val u = gArray(j).u

        val traceQ = trace(u.v * lambdaJ)

        pow(w(i, j), 2) * trace(u.v * lambdaJ)
      }.sum

      val traceQD = 0.5 * traceQTerm

      val tildePD = 0.5 * kTildeDiagSum_i

      val tracePD = 0.5 * trace(v.v * lambdaI)

      dBeta(i) = logTermD - tildeQD - tildePD - traceQD - tracePD
    }

    dBeta
  }
}