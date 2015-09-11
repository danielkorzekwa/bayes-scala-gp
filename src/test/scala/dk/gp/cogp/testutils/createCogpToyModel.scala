package dk.gp.cogp.testutils

import breeze.linalg.DenseMatrix
import dk.gp.cov.CovSEiso
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import breeze.numerics._
import dk.gp.cov.CovNoise
import dk.gp.cogp.model.CogpGPVar
import breeze.stats._
import breeze.linalg._
import dk.gp.cogp.model.CogpModel

object createCogpToyModel {

  def apply(x: DenseMatrix[Double], y: DenseMatrix[Double], z: DenseMatrix[Double]): CogpModel = {

    val gVariable = CogpGPVar(y = mean(y(*, ::)), z, covFunc = CovSEiso(), covFuncParams = DenseVector(log(1), log(1)))

    val hVariable0 = CogpGPVar(y = y(::, 0), z, covFunc = CovNoise(), covFuncParams = DenseVector(log(1)))
    val hVariable1 = CogpGPVar(y = y(::, 1), z, covFunc = CovNoise(), covFuncParams = DenseVector(log(1)))

    //likelihood noise precision
    val beta = DenseVector(1d / 0.01, 1d / 0.01) // [P x 1]

    //mixing weights
    val w = new DenseMatrix(2, 1, Array(1.0, 1)) // [P x Q]

    val model = CogpModel(x, y, z, Array(gVariable), Array(hVariable0, hVariable1), beta, w)
    model
  }

  def apply(x: DenseMatrix[Double], y: DenseMatrix[Double]): CogpModel = createCogpToyModel(x, y, z = x)

}