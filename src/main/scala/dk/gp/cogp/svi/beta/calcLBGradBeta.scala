package dk.gp.cogp.svi.beta

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.InjectNumericOps
import breeze.linalg.diag
import breeze.linalg.sum
import breeze.linalg.trace
import breeze.numerics.pow
import dk.gp.cogp.lb.LowerBound
import dk.gp.cov.utils.covDiag
import dk.gp.cogp.lb.wAm

object calcLBGradBeta {

  def apply(lb: LowerBound, y: DenseMatrix[Double]): DenseVector[Double] = {

    val dBeta = DenseVector.zeros[Double](lb.model.beta.size)
    for (i <- 0 until lb.model.beta.size) {
      dBeta(i) = logTermD(i, lb, y) - 0.5 * kTildeQTermD(i, lb) - 0.5 * kTildeDiagSumi(i, lb) - 0.5 * traceQTermD(i, lb) - tracePD(i, lb)
    }

    dBeta
  }

  private def tracePD(i: Int, lb: LowerBound): Double = {

    val kZZiInv = lb.kZZiInv(i)

    val kXZi = lb.kXZi(i)

    val now = System.currentTimeMillis()
    val Ai = kXZi * kZZiInv //@TODO Ai term is computed many times across different lb derivatives, the same with Aj

    val lambdaI = Ai.t * Ai

    val v = lb.model.h(i).u

    val tracePD = 0.5 * trace(v.v * lambdaI)
    tracePD
  }

  private def logTermD(i: Int, lb: LowerBound, y: DenseMatrix[Double]): Double = {

    val beta = lb.model.beta(i)

    val kZZiInv = lb.kZZiInv(i)

    val kXZi = lb.kXZi(i)

    val Ai = kXZi * kZZiInv

    val yTerm = y(::, i) - wAm(i, lb) - Ai * lb.model.h(i).u.m
    val logTermD = (0.5 * y.rows) / beta - 0.5 * sum(pow(yTerm, 2))

    logTermD
  }

  private def kTildeDiagSumi(i: Int, lb: LowerBound): Double = {
    val kZZiInv = lb.kZZiInv(i)
    val kXZi = lb.kXZi(i)
    val kZXi = kXZi.t

    val kXXiDiag = covDiag(lb.x, lb.model.h(i).covFunc, lb.model.h(i).covFuncParams)

    val kTildeDiagSumi = sum(kXXiDiag - diag(kXZi * kZZiInv * kZXi))

    kTildeDiagSumi
  }

  private def traceQTermD(i: Int, lb: LowerBound) = {
    val traceQTermD = (0 until lb.model.g.size).map { j =>

      val kXZ = lb.kXZj(j)
      val kZZ = lb.kZZj(j)
      val kZZinv = lb.kZZjInv(j)

      val Aj = kXZ * kZZinv
      val lambdaJ = Aj.t * Aj
      val u = lb.model.g(j).u

      val traceQ = trace(u.v * lambdaJ)

      pow(lb.model.w(i, j), 2) * trace(u.v * lambdaJ)
    }.sum

    traceQTermD
  }

  private def kTildeQTermD(i: Int, lb: LowerBound): Double = {

    val kTildeQTerm = (0 until lb.model.g.size).map { j =>

      val kXZ = lb.kXZj(j)
      val kZZ = lb.kZZj(j)
      val kZX = kXZ.t
      val kXXDiag = covDiag(lb.x, lb.model.g(j).covFunc, lb.model.g(j).covFuncParams)

      val kZZinv = lb.kZZjInv(j)
      val kTildeDiagSum = sum(kXXDiag - diag(kXZ * kZZinv * kZX))

      pow(lb.model.w(i, j), 2) * kTildeDiagSum
    }.sum

    kTildeQTerm
  }


}