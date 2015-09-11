package dk.gp.cogp.lb.grad

import breeze.linalg.DenseMatrix
import breeze.linalg.inv
import breeze.numerics._
import dk.gp.cogp.lb.LowerBound
import breeze.linalg.InjectNumericOps
import dk.gp.cogp.model.CogpModel

object calcLBGradUEta2 {

  def apply(j: Int, lowerBound:LowerBound,model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = {

    val kXZ = lowerBound.kXZj(j)
    val kZZ = lowerBound.kZZj(j)

    val kZZinv = lowerBound.kZZjInv(j)
    
    val beta = model.beta
    val w = model.w

    val u = model.g.map(_.u)

    val A = kXZ * kZZinv
    val tmp = (0 until beta.size).map { i =>
      val betaVal = beta(i)
      val w2 = pow(w(i, j), 2)
      betaVal * w2 * A.t * A
    }.reduceLeft((total, x) => total + x)

    val lambda = kZZinv + tmp

   //  val vCholR = cholesky(u(j).v).t
  //  val vInv = invchol(vCholR)
     val vInv = inv(u(j).v) //@TODO use cholesky with jitter here
    val eta2Grad = 0.5 * vInv - 0.5 * lambda

    eta2Grad
  }
}