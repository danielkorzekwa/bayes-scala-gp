package dk.gp.gpr

import breeze.linalg.DenseMatrix
import scala.math._
import breeze.linalg.DenseVector
import breeze.linalg.diag

object predict {

  def apply(z: DenseMatrix[Double], model: GprModel): DenseMatrix[Double] = {
    val kXZ = model.covFunc.cov(model.x, z, model.covFuncParams)

    val kZZ = model.covFunc.cov(z, z, model.covFuncParams) + exp(2 * model.noiseLogStdDev) * DenseMatrix.eye[Double](z.rows)

    val meanZ = DenseVector.zeros[Double](z.rows) + model.mean

    //@TODO use Cholesky Factorization instead of a direct inverse
    val predMean = meanZ + kXZ.t * (model.kXXInv * (model.y - model.meanX))
    val predVar = kZZ - kXZ.t * model.kXXInv * kXZ

    DenseVector.horzcat(predMean, diag(predVar))

  }

}