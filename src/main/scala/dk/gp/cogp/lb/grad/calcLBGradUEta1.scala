package dk.gp.cogp.lb.grad

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.inv
import dk.gp.cogp.CogpModel
import dk.gp.cogp.lb.LowerBound
import breeze.linalg.InjectNumericOps

object calcLBGradUEta1 {

  def apply(j: Int, lowerBound:LowerBound, model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseVector[Double] = {

    val kXZ = lowerBound.kXZj(j)
    val kZZ = lowerBound.kZZj(j)

    val beta = model.beta
    val w = model.w
    val u = model.g.map(_.u)
    val v = model.h.map(_.u)

    val kZZinv = lowerBound.kZZjInv(j)

    val Aj = kXZ * kZZinv

    val tmp = (0 until beta.size).map { i =>
      val betaVal = beta(i)
      val wVal = w(i, j)

      val othersJIdx = (0 until w.cols).filter(jIndex => jIndex != j)
      val wAm = if (othersJIdx.size > 0) {
        othersJIdx.map { jIndex => w(i, jIndex) * Aj * u(jIndex).m }.toArray.sum
      } else DenseVector.zeros[Double](y.rows)


      val Ai = lowerBound.calcAi(i)

      val yVal = y(::, i) - Ai * v(i).m - wAm

      betaVal * wVal * Aj.t * yVal
    }.reduceLeft((total, x) => total + x)

    //@TODO why u(j).v is used here instead of [..] term from eq 19. Computing gradient with respect to natural parameters using chain rule indicates that [...] should be used.
    // Is it because u(j).v=Sj is a maximiser of [....] when setting derivative of Lower bound with respect to Sj to 0?

   // val vCholR = cholesky(u(j).v).t
   // val vInv = invchol(vCholR)
    val vInv = inv(u(j).v) //@TODO use cholesky with jitter here
    val eta1Grad = tmp - vInv * u(j).m 
    eta1Grad
  }

  //@TODO replace it with ufunc sum(.) from breeze
  implicit class DenseVectorOps(seq: Array[DenseVector[Double]]) {
    def sum(): DenseVector[Double] = seq match {
      case Array(x) => x
      case seq      => seq.reduceLeft((total, x) => total + x)
    }
  }
}