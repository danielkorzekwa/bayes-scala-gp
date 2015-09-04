package dk.gp.cogp.svi.hypcovh

import dk.gp.cogp.CogpModel
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.trace
import breeze.linalg.inv
import dk.gp.cov.utils.covDiagD
import breeze.linalg.sum
import breeze.linalg.diag
import breeze.linalg.cholesky
import dk.gp.math.invchol
import dk.gp.cogp.lb.LowerBound

object calcLBGradHypCovH {

  def apply(i: Int, lowerBound: LowerBound, model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseVector[Double] = {

    val hArray = model.h
    val gArray = model.g

    val z = model.h(i).z

    val kXZ = lowerBound.kXZi(i)
    val kZZ = lowerBound.kZZi(i)
    val kZZdArray = model.h(i).covFunc.covD(z, model.h(i).covFuncParams)
    val kXZDArray = model.h(i).covFunc.covD(x, z, model.h(i).covFuncParams)

    val dKxxDiagArray = covDiagD(z, model.h(i).covFunc, model.h(i).covFuncParams)

    val kZZinv = lowerBound.kZZiInv(i)
    val Ai = kXZ * kZZinv
    val kZX = kXZ.t

    val u = model.h(i).u //@TODO name u as u_h or hU
    val beta = model.beta
    val w = model.w

    val covParamsD = (0 until model.h(i).covFuncParams.size).map { k =>

      val dKzz = kZZdArray(k)
      val dKxz = kXZDArray(k)
      val dKxxDiag = dKxxDiagArray(k)

      val dAi = dKxz * kZZinv - Ai * dKzz * kZZinv

      val wAm = (0 until gArray.size).foldLeft(DenseVector.zeros[Double](x.rows)) { (wAm, j) =>

        val kXZ = lowerBound.kXZj(j)
        val kZZ = lowerBound.kZZj(j)
        val kZZinv = lowerBound.kZZjInv(j)

        val Aj = kXZ * kZZinv

        wAm + w(i, j) * Aj * gArray(j).u.m
      }

      val yTerm = y(::, i) - wAm - Ai * hArray(i).u.m
      val logTerm = beta(i) * (yTerm.t * dAi * u.m) //@TODO performance improvement

      val tildeP = 0.5 * beta(i) * sum(dKxxDiag - diag(dAi * kZX) - diag(Ai * dKxz.t)) //@TODO performance improvement
      val traceP = beta(i) * trace(u.v * dAi * Ai)
      val lkl = 0.5d * trace(kZZinv * dKzz) - 0.5 * trace(kZZinv * dKzz * kZZinv * (u.m * u.m.t + u.v))

      logTerm - tildeP - traceP - lkl
    }.toArray

    DenseVector(covParamsD)
  }
}