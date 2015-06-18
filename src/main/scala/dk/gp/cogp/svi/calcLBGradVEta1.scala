package dk.gp.cogp.svi

import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import breeze.linalg.inv
import dk.gp.math.MultivariateGaussian

object calcLBGradVEta1 {

  def apply(beta: Double, w: DenseVector[Double], kZZ: DenseMatrix[Double], kXZ: DenseMatrix[Double], S: DenseMatrix[Double], y: DenseVector[Double], mP: DenseVector[Double], u: Array[MultivariateGaussian]): DenseVector[Double] = {

    val A = kXZ * inv(kZZ)

    val wam = if (w.size == 0) DenseVector.zeros[Double](y.size)
    else {
      (0 until w.size).map(jIndex => w(jIndex) * A * u(jIndex).m)
    }.toArray.sum

    val yVal = y - wam

    val grad = beta * A.t * yVal - inv(S) * mP

    grad
  }

  implicit class DenseVectorOps(seq: Array[DenseVector[Double]]) {
    def sum(): DenseVector[Double] = seq match {
      case Array(x) => x
      case seq      => seq.reduceLeft((total, x) => total + x)
    }
  }
}