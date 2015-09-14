package dk.gp.cogp.lb.grad

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.trace
import dk.gp.cov.utils.covDiagD
import breeze.numerics._
import breeze.linalg._
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.wAm

object calcLBGradHypCovG {

  def apply(j: Int, lb: LowerBound, y: DenseMatrix[Double]): DenseVector[Double] = {

    val x = lb.x
    val z = lb.model.g(j).z
    val kZZdArray = lb.model.g(j).covFunc.covD(z, z, lb.model.g(j).covFuncParams)
    val kXZDArray = lb.model.g(j).covFunc.covD(x, z, lb.model.g(j).covFuncParams)

    val kZZinv = lb.kZZjInv(j)
    val Aj = lb.calcAj(j)

    val kXXDiagDArray = covDiagD(x, lb.model.g(j).covFunc, lb.model.g(j).covFuncParams)

    val covParamsD = lb.model.g(j).covFuncParams.mapPairs { (k, param) =>

      val kZZd = kZZdArray(k)
      val kXZd = kXZDArray(k)
      val kXXDiagD = kXXDiagDArray(k)
      val AjD = kXZd * kZZinv - Aj * kZZd * kZZinv

      logTermPart(j, lb, y, AjD) - tildeQPart(j, lb, AjD, kXXDiagD, kXZd) - traceQPart(j, lb, AjD) - lklPart(j, lb, kZZd)

    }.toArray

    DenseVector(covParamsD)
  }

  private def logTermPart(j: Int, lb: LowerBound, y: DenseMatrix[Double], AjD: DenseMatrix[Double]): Double = {

    val beta = lb.model.beta
    val w = lb.model.w
    val u = lb.model.g(j).u

    val logTermPart = (0 until lb.model.h.size).map { i =>

      val Ai = lb.calcAi(i)
      val yTerm = y(::, i) - wAm(i, lb) - Ai * lb.model.h(i).u.m
      beta(i) * w(i, j) * (yTerm.t * AjD * u.m)

    }.sum

    logTermPart
  }

  private def lklPart(j: Int, lb: LowerBound, kZZd: DenseMatrix[Double]): Double = {

    val kZZinv = lb.kZZjInv(j)
    val u = lb.model.g(j).u

    0.5 * trace(kZZinv * kZZd) - 0.5 * trace(kZZinv * kZZd * kZZinv * (u.m * u.m.t + u.v))
  }

  private def tildeQPart(j: Int, lb: LowerBound, AjD: DenseMatrix[Double], kXXDiagD: DenseVector[Double], kXZd: DenseMatrix[Double]): Double = {
    val beta = lb.model.beta
    val w = lb.model.w
    val Aj = lb.calcAj(j)

    val kZX = lb.kXZj(j).t

    val tildeQPart = (0 until lb.model.h.size).map { i =>

      val tilde = 0.5 * beta(i) * pow(w(i, j), 2) * sum(kXXDiagD - diag(AjD * kZX) - diag(Aj * kXZd.t))
      tilde

    }.sum

    tildeQPart
  }

  private def traceQPart(j: Int, lb: LowerBound, AjD: DenseMatrix[Double]): Double = {

    val w = lb.model.w
    val beta = lb.model.beta
    val u = lb.model.g(j).u

    val kXZ = lb.kXZj(j)
    val kZZinv = lb.kZZjInv(j)
    val Aj = lb.calcAj(j)

    val traceQPart = (0 until lb.model.h.size).map { i =>
      beta(i) * trace(pow(w(i, j), 2) * u.v * (AjD.t * Aj)) //@TODO performance improvement
    }.sum

    traceQPart
  }
}