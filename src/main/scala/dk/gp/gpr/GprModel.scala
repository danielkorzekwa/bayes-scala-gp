package dk.gp.gpr

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import scala.math._
import breeze.linalg.inv
import breeze.linalg.cholesky
import dk.gp.math.invchol

//@TODO remove x and y from the model, keep covFunc and covFuncParams only
case class GprModel(x: DenseMatrix[Double], y: DenseVector[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], noiseLogStdDev: Double, mean: Double = 0d) {

  val kXX = covFunc.cov(x, x, covFuncParams) + exp(2 * noiseLogStdDev) * DenseMatrix.eye[Double](x.rows) + DenseMatrix.eye[Double](x.rows) * 1e-7
  val kXXInv = invchol(cholesky(kXX).t)

  val meanX = DenseVector.zeros[Double](x.rows) + mean

}