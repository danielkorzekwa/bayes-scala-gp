package dk.gp.cogp.svi.u

import breeze.linalg.DenseMatrix
import breeze.linalg.inv
import breeze.linalg.DenseVector
import breeze.numerics._
import dk.gp.cogp.CogpModel
import dk.gp.cogp.CogpModel
import breeze.linalg.cholesky
import dk.gp.math.invchol

object calcLBGradUEta2 {

  def apply(j: Int, model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = {

    val z = model.g(j).z
    val kXZ = model.g(j).covFunc.cov(z, z, model.g(j).covFuncParams)
    val kZZ = model.g(j).covFunc.cov(z, z, model.g(j).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)

    val kZZCholR = cholesky(kZZ).t
    val kZZinv = invchol(kZZCholR)
    
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