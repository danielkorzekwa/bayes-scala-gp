package dk.gp.cogp.testutils

import dk.gp.cogp.CogpModel
import breeze.linalg.DenseMatrix
import dk.gp.cov.CovSEiso
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import breeze.numerics._

object createCogpModel {

  def apply(x: DenseMatrix[Double],y:DenseMatrix[Double]): CogpModel = {
  
    val covFuncG: Array[CovFunc] = Array(CovSEiso())
    val cofFuncGParams = Array(DenseVector(log(1), log(1)))

    val covFuncH: Array[CovFunc] = Array(CovSEiso(), CovSEiso())
    val covFuncHParams = Array(DenseVector(log(1), log(1e-10)), DenseVector(log(1), log(1e-10)))

    val model = CogpModel(x, y, covFuncG, cofFuncGParams, covFuncH, covFuncHParams)

    model
  }
}