package dk.gp.cogp.svi.u

import breeze.linalg.DenseMatrix
import breeze.linalg.inv
import breeze.linalg.DenseVector
import breeze.numerics._
import dk.gp.cogp.svi.LBState

object calcLBGradUEta2 {

  def apply(j: Int, beta: DenseVector[Double], w: DenseMatrix[Double],  kZZ: DenseMatrix[Double], kXZ: DenseMatrix[Double],
      lbState:LBState):DenseMatrix[Double] = {
      val u = lbState.u
    
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