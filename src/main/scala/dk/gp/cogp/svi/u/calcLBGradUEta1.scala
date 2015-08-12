package dk.gp.cogp.svi.u

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.inv
import dk.gp.cogp.CogpModel

object calcLBGradUEta1 {

  def apply(j: Int, model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseVector[Double] = {

    val z = model.g(j).z
    val kXZ = model.g(j).covFunc.cov(z, z, model.g(j).covFuncParams)
    val kZZ = model.g(j).covFunc.cov(z, z, model.g(j).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)

    val beta = model.beta
    val w = model.w
    val u = model.g.map(_.u)
    val v = model.h.map(_.u)

    val Aj = kXZ * inv(kZZ)

    val tmp = (0 until beta.size).map { i =>
      val betaVal = beta(i)
      val wVal = w(i, j)

      val othersJIdx = (0 until w.cols).filter(jIndex => jIndex != j)
      val wAm = if (othersJIdx.size > 0) {
        othersJIdx.map { jIndex => w(i, jIndex) * Aj * u(jIndex).m }.toArray.sum
      } else DenseVector.zeros[Double](y.rows)

      val z = model.h(i).z
      val kZZ2 = model.h(i).covFunc.cov(z, z, model.h(i).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)
      val kXZ2 = model.h(i).covFunc.cov(z, z, model.h(i).covFuncParams)
      val kZX2 = kXZ2.t
      val Ai = kXZ2 * inv(kZZ2)

      val yVal = y(::, i) - Ai * v(i).m - wAm

      betaVal * wVal * Aj.t * yVal
    }.reduceLeft((total, x) => total + x)

    //@TODO why u(j).v is used here instead of [..] term from eq 19. Computing gradient with respect to natural parameters using chain rule indicates that [...] should be used.
    // Is it because u(j).v=Sj is a maximiser of [....] when setting derivative of Lower bound with respect to Sj to 0?
    val eta1Grad = tmp - inv(u(j).v) * u(j).m
    eta1Grad
  }

  implicit class DenseVectorOps(seq: Array[DenseVector[Double]]) {
    def sum(): DenseVector[Double] = seq match {
      case Array(x) => x
      case seq      => seq.reduceLeft((total, x) => total + x)
    }
  }
}