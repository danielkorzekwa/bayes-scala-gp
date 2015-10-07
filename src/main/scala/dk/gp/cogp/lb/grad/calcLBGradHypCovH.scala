package dk.gp.cogp.lb.grad

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.trace
import dk.gp.cov.utils.covDiagD
import breeze.linalg.sum
import breeze.linalg.diag
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.wAm
import dk.gp.math.diagProd

object calcLBGradHypCovH {

  def apply(i: Int, lb: LowerBound): DenseVector[Double] = {

    val model = lb.model

    val kXZ = lb.kXZi(i)
    val kZZ = lb.kZZi(i)

    val kZZinv = lb.kZZiInv(i)
    val Ai = lb.calcAi(i)

    val u = model.h(i).u //@TODO name u as u_h or hU
    val beta = model.beta
    val w = model.w

    val covParamsD = (0 until model.h(i).covFuncParams.size).map { k =>

      val dKzz = lb.dKzzi(i, k)
      val dKxz = lb.dKxzi(i, k)
      val dKxxDiag = lb.calcdKxxDiagi(i, k)

      val dAi = lb.calcdAi(i, k)

      val y = lb.yi(i)
      val yTerm = y - wAm(i, lb) - Ai * model.h(i).u.m
      val logTerm = beta(i) * (yTerm.t * dAi * u.m) //@TODO performance improvement

      val tildeP = 0.5 * beta(i) * sum(dKxxDiag - diagProd(dAi,kXZ) - diagProd(Ai, dKxz))
      val traceP = beta(i) * sum(diagProd(Ai*u.v,dAi))
      val lkl = 0.5d * trace(kZZinv * dKzz) - 0.5 * trace(kZZinv * dKzz * kZZinv * (u.m * u.m.t + u.v))

      logTerm - tildeP - traceP - lkl
    }.toArray

    DenseVector(covParamsD)
  }
}