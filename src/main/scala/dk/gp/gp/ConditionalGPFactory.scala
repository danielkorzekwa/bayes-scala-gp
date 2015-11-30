package dk.gp.gp

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import breeze.linalg.cholesky
import dk.gp.math.invchol
import breeze.linalg._

case class ConditionalGPFactory(x: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], mean: Double = 0) {

  val kXX = covFunc.cov(x, x, covFuncParams) + DenseMatrix.eye[Double](x.rows) * 1e-7
  val meanX = DenseVector.zeros[Double](x.rows) + mean

  val lXX = cholesky(kXX).t
  val kXXinv = invchol(lXX)
  val lXXinv = inv(lXX)

  /**
   * @return A,b,v parts of Linear Conditional Gaussian p(t|x) = N(Ax+b,v)
   */
  def create(t: DenseMatrix[Double]): Tuple3[DenseMatrix[Double], DenseVector[Double], DenseMatrix[Double]] = {

    val kTT = covFunc.cov(t, t, covFuncParams) + DenseMatrix.eye[Double](t.rows) * 1e-7
    val meanT = DenseVector.zeros[Double](t.rows) + mean

    val kTX = covFunc.cov(t, x, covFuncParams)

    val A = kTX * kXXinv
    val b = if(mean==0d) DenseVector.zeros[Double](t.rows) else {meanT - A * meanX}
    val kTXInvLXX = kTX * lXXinv
    val v = kTT - kTXInvLXX * kTXInvLXX.t
    (A, b, v)

  }
}