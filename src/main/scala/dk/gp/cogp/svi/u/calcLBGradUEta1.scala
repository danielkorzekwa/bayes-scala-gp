package dk.gp.cogp.svi.u

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cogp.model.CogpModelParams
import breeze.linalg.inv
import dk.gp.cogp.svi.LBState

object calcLBGradUEta1 {

  def apply(j: Int, beta: DenseVector[Double], w: DenseMatrix[Double], y: DenseMatrix[Double],
            kZZ: DenseMatrix[Double], kXZ: DenseMatrix[Double], lbState: LBState): DenseVector[Double] = {

    val u = lbState.u
    val v = lbState.v

    val A = kXZ * inv(kZZ)

    val tmp = (0 until beta.size).map { i =>
      val betaVal = beta(i)
      val wVal = w(i, j)

      val othersJIdx = (0 until w.cols).filter(jIndex => jIndex != j)
      val wAm = if (othersJIdx.size > 0) {
        othersJIdx.map { jIndex => w(i, jIndex) * A * u(jIndex).m }.toArray.sum
      } else DenseVector.zeros[Double](y.rows)

      val yVal = y(::, i) - A * v(i).m - wAm

      betaVal * wVal * A.t * yVal
    }.reduceLeft((total, x) => total + x)

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