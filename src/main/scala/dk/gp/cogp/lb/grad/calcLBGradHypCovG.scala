package dk.gp.cogp.lb.grad

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.trace
import dk.gp.cov.utils.covDiagD
import breeze.numerics._
import breeze.linalg._
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.wAm
import dk.gp.math.diagProd

object calcLBGradHypCovG {

  def apply(j: Int, lb: LowerBound): DenseVector[Double] = {

    val covParamsD = lb.model.g(j).covFuncParams.mapPairs { (k, param) =>
      val kZZd = lb.dKzzj(j, k)
      logTermPart(j, k, lb) - tildeQPart(j, k, lb) - traceQPart(j, k, lb) - lklPart(j, lb, kZZd)

    }.toArray

    DenseVector(covParamsD)
  }

  private def logTermPart(j: Int, k: Int, lb: LowerBound): Double = {

    val beta = lb.model.beta
    val w = lb.model.w
    val u = lb.model.g(j).u

    val logTermPart = (0 until lb.model.h.size).map { i =>

      val AjD = lb.calcdAj(i, j, k)
      val Ai = lb.calcAi(i)
      val y = lb.yi(i)

      val yTerm = y - wAm(i, lb) - Ai * lb.model.h(i).u.m
      beta(i) * w(i, j) * (yTerm.t * AjD * u.m)

    }.sum

    logTermPart
  }

  private def lklPart(j: Int, lb: LowerBound, kZZd: DenseMatrix[Double]): Double = {

    val kZZinv = lb.kZZjInv(j)
    val u = lb.model.g(j).u

    0.5 * trace(kZZinv * kZZd) - 0.5 * trace(kZZinv * kZZd * kZZinv * (u.m * u.m.t + u.v))
  }

  private def tildeQPart(j: Int, k: Int, lb: LowerBound): Double = {
    val beta = lb.model.beta
    val w = lb.model.w

    val tildeQPart = (0 until lb.model.h.size).map { i =>

      val kXZ = lb.kXZj(i, j)
      val dKxz = lb.dKxzj(i, j, k)

      val AjD = lb.calcdAj(i, j, k)
      val Aj = lb.Aj(i, j)

      val dKxxDiag = lb.calcdKxxDiagj(i, j, k)
      val tilde = 0.5 * beta(i) * pow(w(i, j), 2) * sum(dKxxDiag - diagProd(AjD, kXZ) - diagProd(Aj, dKxz))
      tilde

    }.sum

    tildeQPart
  }

  private def traceQPart(j: Int, k: Int, lb: LowerBound): Double = {

    val w = lb.model.w
    val beta = lb.model.beta
    val u = lb.model.g(j).u

    val traceQPart = (0 until lb.model.h.size).map { i =>

      val AjD = lb.calcdAj(i, j, k)
      val Aj = lb.Aj(i, j)
      beta(i) * pow(w(i, j), 2) * sum(diagProd(Aj * u.v, AjD))
    }.sum

    traceQPart
  }
}