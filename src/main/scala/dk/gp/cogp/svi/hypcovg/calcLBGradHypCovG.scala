package dk.gp.cogp.svi.hypcovg

import dk.gp.cogp.CogpModel
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.trace
import breeze.linalg.inv
import dk.gp.cov.utils.covDiagD
import breeze.numerics._
import breeze.linalg._

object calcLBGradHypCovG {

  def apply(j: Int, model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseVector[Double] = {

    val hArray = model.h
    val gArray = model.g

    val z = model.g(j).z
    val kXZ = model.g(j).covFunc.cov(x, z, model.g(j).covFuncParams)
    val kZZ = model.g(j).covFunc.cov(z, z, model.g(j).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)
    val kZZdArray = model.g(j).covFunc.covD(z, model.g(j).covFuncParams)
    val kXZDArray = model.g(j).covFunc.covD(x, z, model.g(j).covFuncParams)

    val Aj = kXZ * inv(kZZ)
    val kZX = kXZ.t

    val kXXDiagDArray = covDiagD(z, model.g(j).covFunc, model.g(j).covFuncParams)

    val beta = model.beta
    val w = model.w

    val u = model.g(j).u

    val covParamsD = (0 until model.g(j).covFuncParams.size).map { k =>

      val kZZd = kZZdArray(k)
      val kXZd = kXZDArray(k)

      val kXXDiagD = kXXDiagDArray(k)

      val AjD = kXZd * inv(kZZ) - Aj * kZZd * inv(kZZ)

      val logTermPart = (0 until hArray.size).map { i =>

        val z = model.h(i).z
        val kZZ2 = model.h(i).covFunc.cov(z, z, model.h(i).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)
        val kXZ2 = model.h(i).covFunc.cov(z, z, model.h(i).covFuncParams)
        val kZX2 = kXZ2.t
        val Ai2 = kXZ2 * inv(kZZ2)

        val wAm = (0 until gArray.size).foldLeft(DenseVector.zeros[Double](x.rows)) { (wAm, j) =>

          val z = model.g(j).z
          val kXZ = model.g(j).covFunc.cov(x, z, model.g(j).covFuncParams)
          val kZZ = model.g(j).covFunc.cov(z, z, model.g(j).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)
          val Aj = kXZ * inv(kZZ)

          wAm + w(i, j) * Aj * gArray(j).u.m
        }

        val yTerm = y(::, i) - wAm - Ai2 * hArray(i).u.m

        beta(i) * w(i, j) * (yTerm.t * AjD * u.m) //@TODO performance improvement

      }.sum

      val tildeQPart = (0 until hArray.size).map { i =>
        val tilde = 0.5 * beta(i) * pow(w(i, j), 2) * sum(kXXDiagD - diag(AjD * kZX) - diag(Aj * kXZd.t))
        tilde

      }.sum

      val traceQPart = (0 until hArray.size).map { i =>
        0.5 * beta(i) * trace(pow(w(i, j), 2) * u.v * (AjD.t * Aj + Aj.t * AjD)) //@TODO performance improvement
      }.sum

      val lklPart = 0.5 * trace(inv(kZZ) * kZZd) - 0.5 * trace(inv(kZZ) * kZZd * inv(kZZ) * (u.m * u.m.t + u.v))

      logTermPart - tildeQPart - traceQPart - lklPart

    }.toArray

    DenseVector(covParamsD)
  }
}