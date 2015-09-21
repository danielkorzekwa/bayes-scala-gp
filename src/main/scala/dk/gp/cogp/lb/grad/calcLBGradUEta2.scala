package dk.gp.cogp.lb.grad

import breeze.linalg.DenseMatrix
import breeze.linalg.inv
import breeze.numerics._
import dk.gp.cogp.lb.LowerBound
import breeze.linalg.InjectNumericOps
import dk.gp.cogp.model.CogpModel
import breeze.linalg.cholesky
import dk.gp.math.invchol

object calcLBGradUEta2 {

  def apply(j: Int, lb: LowerBound): DenseMatrix[Double] = {

    val kZZinv = lb.kZZjInv(j)
    val beta = lb.model.beta
    val w = lb.model.w
    val g = lb.model.g

    val tmp = (0 until beta.size).map { i =>

      val Aj = lb.calcAj(i, j)
      beta(i) * pow(w(i, j), 2) * Aj.t * Aj
    }.reduceLeft((total, x) => total + x)

    val lambda = kZZinv + tmp

    val vInv = invchol(cholesky(lb.model.g(j).u.v).t)
    val eta2Grad = 0.5 * vInv - 0.5 * lambda

    eta2Grad
  }
}