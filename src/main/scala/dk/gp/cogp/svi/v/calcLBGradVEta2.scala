package dk.gp.cogp.svi.v

import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import breeze.linalg.inv

object calcLBGradVEta2 {

  def apply(beta: Double, S: DenseMatrix[Double], kZZ: DenseMatrix[Double], kXZ: DenseMatrix[Double]): DenseMatrix[Double] = {

    val A = kXZ * inv(kZZ)

    val lambda = inv(kZZ) + beta * A.t * A

    val grad = 0.5 * inv(S) - 0.5 * lambda

    grad
  }

}