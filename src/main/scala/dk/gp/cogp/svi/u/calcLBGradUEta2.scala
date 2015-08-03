package dk.gp.cogp.svi.u

import breeze.linalg.DenseMatrix
import breeze.linalg.inv
import breeze.linalg.DenseVector
import breeze.numerics._
import dk.gp.cogp.CogpModel
import dk.gp.cogp.CogpModel

object calcLBGradUEta2 {

  def apply(j: Int, beta: DenseVector[Double], w: DenseMatrix[Double],  kZZ: DenseMatrix[Double], kXZ: DenseMatrix[Double],
      model:CogpModel):DenseMatrix[Double] = {
    
    val u = model.g.map(_.u)
    
    val A = kXZ * inv(kZZ)
    val tmp = (0 until beta.size).map { i =>
      val betaVal = beta(i)
      val w2 = pow(w(i, j), 2)
      betaVal * w2 * A.t * A
    }.reduceLeft((total, x) => total + x)

    val lambda = inv(kZZ) + tmp
    
    val eta2Grad = 0.5 * inv(u(j).v) - 0.5 * lambda
  
    eta2Grad
  }
}