package dk.gp.gpr

import breeze.linalg.DenseMatrix
import scala.math._
import breeze.linalg.DenseVector
import breeze.linalg.diag

object gprPredict {

  def apply(z: DenseMatrix[Double], model: GprModel): DenseMatrix[Double] = {

    val meanX = model.meanFunc(model.x)

    val kXXInv = {
      val kXX = model.calcKXX()
      val kXXInv = model.calcKXXInv(kXX)
      kXXInv
    }
    val t1 = (kXXInv * (model.y - meanX))

    val kXZ = model.covFunc.cov(model.x, z, model.covFuncParams)

    val kZZ = model.covFunc.cov(z, z, model.covFuncParams) + exp(2 * model.noiseLogStdDev) * DenseMatrix.eye[Double](z.rows)

    val meanZ = model.meanFunc(z)

    val predMean = meanZ + kXZ.t * (kXXInv * (model.y - meanX))
    val predVar = kZZ - kXZ.t * kXXInv * kXZ

    DenseVector.horzcat(predMean, diag(predVar))

  }

}