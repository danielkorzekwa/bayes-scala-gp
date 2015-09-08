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
import dk.gp.cogp.lb.wAm

object calcLBGradHypCovH {

  def apply(i: Int, lb: LowerBound,  y: DenseMatrix[Double]): DenseVector[Double] = {

    val model = lb.model
    val x = lb.x
    
    val hArray = model.h
    val gArray = model.g

    val z = model.h(i).z

    val kXZ = lb.kXZi(i)
    val kZZ = lb.kZZi(i)
    val kZZdArray = model.h(i).covFunc.covD(z,z, model.h(i).covFuncParams)
    val kXZDArray = model.h(i).covFunc.covD(x, z, model.h(i).covFuncParams)

    val dKxxDiagArray = covDiagD(x, model.h(i).covFunc, model.h(i).covFuncParams)

    val kZZinv = lb.kZZiInv(i)
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

    
      val yTerm = y(::, i) - wAm(i,lb) - Ai * hArray(i).u.m
      val logTerm = beta(i) * (yTerm.t * dAi * u.m) //@TODO performance improvement

      val tildeP = 0.5 * beta(i) * sum(dKxxDiag - diag(dAi * kZX) - diag(Ai * dKxz.t)) //@TODO performance improvement
      val traceP = beta(i) * trace(u.v * dAi.t * Ai)
      val lkl = 0.5d * trace(kZZinv * dKzz) - 0.5 * trace(kZZinv * dKzz * kZZinv * (u.m * u.m.t + u.v))

      logTerm - tildeP - traceP - lkl
    }.toArray

    DenseVector(covParamsD)
  }
}