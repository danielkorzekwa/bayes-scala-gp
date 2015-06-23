package dk.gp.gpr

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import scala.math._
import breeze.linalg.inv

case class GprModel(x: DenseMatrix[Double], y: DenseVector[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], noiseLogStdDev: Double, mean: Double = 0d) {

  val kXX = covFunc.cov(x, x, covFuncParams) + exp(2 * noiseLogStdDev) * DenseMatrix.eye[Double](x.rows) + DenseMatrix.eye[Double](x.rows) * 1e-7
  val kXXInv = inv(kXX)

  val meanX = DenseVector.zeros[Double](x.rows) + mean

}