package dk.gp.cogp.svi.hypcovg

import dk.gp.cogp.CogpModel
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.trace
import breeze.linalg.inv
import dk.gp.cov.utils.covDiagD

object calcLBGradHypCovG {

  def apply(j: Int, model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseVector[Double] = {

    val z = model.g(j).z
    val kZZ = model.g(j).covFunc.cov(z, z, model.g(j).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)
    val kZZd = model.g(j).covFunc.covD(z, model.g(j).covFuncParams)
    val kZZdArray = model.g(j).covFunc.covD(z, model.g(j).covFuncParams)

    val kXXDiagDArray = covDiagD(z, model.g(j).covFunc, model.g(j).covFuncParams)
    
    val u = model.g(j).u

    val covParamsD = (0 until model.g(j).covFuncParams.size).map { k =>

      val kZZd = kZZdArray(k)

      val lklPart = 0.5 * trace(inv(kZZ) * kZZd) - 0.5 * trace(inv(kZZ) * kZZd * inv(kZZ) * (u.m * u.m.t + u.v))

      -lklPart

    }.toArray

    DenseVector(covParamsD)
  }
}