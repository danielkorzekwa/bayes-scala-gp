package dk.gp.cogp.lb.grad

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.inv
import dk.gp.cogp.lb.LowerBound
import breeze.linalg.InjectNumericOps
import dk.gp.cogp.model.CogpModel
import breeze.numerics._
import breeze.linalg._
import dk.gp.math.invchol

object calcLBGradUEta1 {

  def apply(j: Int, lb: LowerBound): DenseVector[Double] = {

    val model = lb.model
    val beta = model.beta
    val w = model.w
    val u = model.g.map(_.u)
    val v = model.h.map(_.u)

    val tmp = (0 until beta.size).map { i =>
      val betaVal = beta(i)
      val wVal = w(i, j)
      val y = lb.yi(i)

      val Aj = lb.calcAj(i, j)

      val othersJIdx = (0 until w.cols).filter(jIndex => jIndex != j)
      val wAm = if (othersJIdx.size > 0) {
        sum(othersJIdx.map { jIndex => w(i, jIndex) * Aj * u(jIndex).m })
      } else DenseVector.zeros[Double](y.size)

      val Ai = lb.calcAi(i)

      val yVal = y - Ai * v(i).m - wAm

      betaVal * wVal * Aj.t * yVal
    }.reduceLeft((total, x) => total + x)

    //@TODO why u(j).v is used here instead of [..] term from eq 19. Computing gradient with respect to natural parameters using chain rule indicates that [...] should be used.
    // Is it because u(j).v=Sj is a maximiser of [....] when setting derivative of Lower bound with respect to Sj to 0?
     val vInv = invchol( cholesky(u(j).v).t)
    val eta1Grad = tmp - vInv * u(j).m
    eta1Grad
  }

}