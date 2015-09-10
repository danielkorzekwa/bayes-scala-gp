package dk.gp.cogp.lb.grad

import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import dk.gp.cogp.CogpModel
import dk.gp.math.invchol
import breeze.linalg.cholesky
import dk.gp.cogp.lb.LowerBound
import breeze.linalg.InjectNumericOps

object calcLBGradVEta1 {

  def apply(i: Int, lowerBound: LowerBound, model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseVector[Double] = {

    val Ai = lowerBound.calcAi(i)

    val w = model.w
    val beta = model.beta

    val u = model.g.map(_.u)
    val v = model.h.map(_.u)

    val wam = if (w.size == 0) DenseVector.zeros[Double](y.size)
    else {
      (0 until w.cols).map { jIndex =>

        val kXZ = lowerBound.kXZj(jIndex)
        val kZZ = lowerBound.kZZj(jIndex)

        val kZZinv = lowerBound.kZZjInv(jIndex)

        val Aj = kXZ * kZZinv

        w(i, jIndex) * Aj * u(jIndex).m
      }
    }.toArray.sum

    val yVal = y(::, i) - wam

    val vCholR = cholesky(v(i).v).t
    val vInv = invchol(vCholR)

    val grad = beta(i) * Ai.t * yVal - vInv * v(i).m

    grad
  }

  implicit class DenseVectorOps(seq: Array[DenseVector[Double]]) {
    def sum(): DenseVector[Double] = seq match { //@TODO this is not required anymorre, David Hall added ufunc for sum(.) to breeze
      case Array(x) => x
      case seq      => seq.reduceLeft((total, x) => total + x)
    }
  }
}