package dk.gp.cogp.svi.v

import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import breeze.linalg.inv
import dk.gp.math.MultivariateGaussian
import dk.gp.cogp.CogpModel

object calcLBGradVEta1 {

  def apply(i: Int, model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseVector[Double] = {

    val z_i = model.h(i).z
    val kZZ_i = model.h(i).covFunc.cov(z_i, z_i, model.h(i).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)
    val kXZ_i = model.h(i).covFunc.cov(z_i, z_i, model.h(i).covFuncParams)

    val Ai = kXZ_i * inv(kZZ_i)

    val w = model.w
    val beta = model.beta

    val u = model.g.map(_.u)
    val v = model.h.map(_.u)

    val wam = if (w.size == 0) DenseVector.zeros[Double](y.size)
    else {
      (0 until w.cols).map { jIndex =>

        val z = model.g(jIndex).z
        val kXZ = model.g(jIndex).covFunc.cov(z, z, model.g(jIndex).covFuncParams)
        val kZZ = model.g(jIndex).covFunc.cov(z, z, model.g(jIndex).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)

        val Aj = kXZ * inv(kZZ)

        w(i, jIndex) * Aj * u(jIndex).m
      }
    }.toArray.sum

    val yVal = y(::, i) - wam

    val grad = beta(i) * Ai.t * yVal - inv(v(i).v) * v(i).m

    grad
  }

  implicit class DenseVectorOps(seq: Array[DenseVector[Double]]) {
    def sum(): DenseVector[Double] = seq match {
      case Array(x) => x
      case seq      => seq.reduceLeft((total, x) => total + x)
    }
  }
}