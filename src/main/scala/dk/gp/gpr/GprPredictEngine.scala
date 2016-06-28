package dk.gp.gpr

import breeze.linalg.DenseMatrix
import breeze.linalg.diag
import breeze.linalg.DenseVector
import breeze.numerics._
import breeze.linalg._

case class GprPredictEngine(model: GprModel) {

  private val meanX = model.meanFunc(model.x)

  private val kXXInv = {
    val kXX = model.calcKXX()
    val kXXInv = model.calcKXXInv(kXX)
    kXXInv
  }
  private val t1 = (kXXInv * (model.y - meanX))

  def predict(z: DenseMatrix[Double]): DenseMatrix[Double] = {
    val kXZ = model.covFunc.cov(model.x, z, model.covFuncParams)

    val kZZ = model.covFunc.cov(z, z, model.covFuncParams) + exp(2 * model.noiseLogStdDev) * DenseMatrix.eye[Double](z.rows)

    val meanZ = model.meanFunc(z)

    val predMean = meanZ + kXZ.t * t1
    val predVar = kZZ - kXZ.t * kXXInv * kXZ

    DenseVector.horzcat(predMean, diag(predVar))

  }

  def predictMean(z: DenseMatrix[Double]): DenseVector[Double] = {
    val kXZ = model.covFunc.cov(model.x, z, model.covFuncParams)

    val kZZ = model.covFunc.cov(z, z, model.covFuncParams) + exp(2 * model.noiseLogStdDev) * DenseMatrix.eye[Double](z.rows)

    val meanZ = model.meanFunc(z)

    val predMean = meanZ + kXZ.t * t1

    predMean

  }
}