package dk.gp.cogp.lb.grad

import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import dk.gp.math.invchol
import breeze.linalg.cholesky
import dk.gp.cogp.lb.LowerBound
import breeze.linalg.InjectNumericOps
import dk.gp.cogp.model.CogpModel
import breeze.linalg._

object calcLBGradVEta1 {

  def apply(i: Int, lb: LowerBound): DenseVector[Double] = {

    val model = lb.model
    val Ai = lb.calcAi(i)

    val w = model.w
    val beta = model.beta

    val u = model.g.map(_.u)
    val v = model.h.map(_.u)

    val y = lb.yi(i)
    val wam = if (w.size == 0) DenseVector.zeros[Double](y.size)
    else sum{
      (0 until w.cols).map { jIndex =>

        val Aj = lb.calcAj(i,jIndex)

        w(i, jIndex) * Aj * u(jIndex).m
      }
    }

    val yVal = y - wam

    val vCholR = cholesky(v(i).v).t
    val vInv = invchol(vCholR)

    val grad = beta(i) * Ai.t * yVal - vInv * v(i).m

    grad
  }

  
}