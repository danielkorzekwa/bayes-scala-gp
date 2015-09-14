package dk.gp.cogp.lb.grad

import breeze.linalg.DenseMatrix
import breeze.linalg.inv
import breeze.numerics._
import dk.gp.cogp.lb.LowerBound
import breeze.linalg.InjectNumericOps
import dk.gp.cogp.model.CogpModel

object calcLBGradUEta2 {

  def apply(j: Int, lb: LowerBound, y: DenseMatrix[Double]): DenseMatrix[Double] = {

    val model = lb.model
    val kZZinv = lb.kZZjInv(j)

    val beta = model.beta
    val w = model.w

    val u = model.g.map(_.u)

    val Aj = lb.calcAj(j)

    val tmp = (0 until beta.size).map { i =>
      beta(i) * pow(w(i, j), 2) * Aj.t * Aj
    }.reduceLeft((total, x) => total + x)

    val lambda = kZZinv + tmp

    //  val vCholR = cholesky(u(j).v).t
    //  val vInv = invchol(vCholR)
    val vInv = inv(u(j).v) //@TODO use cholesky with jitter here
    val eta2Grad = 0.5 * vInv - 0.5 * lambda

    eta2Grad
  }
}