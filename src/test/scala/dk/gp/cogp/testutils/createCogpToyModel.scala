package dk.gp.cogp.testutils

import dk.gp.cogp.CogpModel
import breeze.linalg.DenseMatrix
import dk.gp.cov.CovSEiso
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import breeze.numerics._
import dk.gp.cov.CovNoise

object createCogpToyModel {

  def apply(x: DenseMatrix[Double], y: DenseMatrix[Double], z: DenseMatrix[Double]): CogpModel = {

    val covFuncG: Array[CovFunc] = Array(CovSEiso())
    val cofFuncGParams = Array(DenseVector(log(1), log(1)))

    val covFuncH: Array[CovFunc] = Array(CovNoise(), CovNoise())
    val covFuncHParams = Array(DenseVector(log(1)), DenseVector(log(1)))

    val model = CogpModel(x, y, z, covFuncG, cofFuncGParams, covFuncH, covFuncHParams)

    model
  }

  def apply(x: DenseMatrix[Double], y: DenseMatrix[Double]): CogpModel = createCogpToyModel(x, y, z = x)

}