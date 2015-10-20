package dk.gp.sgpr

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import breeze.optimize.LBFGS

object sgprTrain {

  def apply(x: DenseMatrix[Double], y: DenseVector[Double], u: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], logNoiseStdDev: Double): SgprModel = {

    val initialParams = DenseVector(covFuncParams.toArray :+ logNoiseStdDev)
    val diffFunction = SparseGpDiffFunction(x, y, u, covFunc)

    val optimizer = new LBFGS[DenseVector[Double]](maxIter = 100, m = 3, tolerance = 1.0E-6)
    val optIterations = optimizer.iterations(diffFunction, initialParams).toList

    val newParams = optIterations.last.x

    val newCovFuncParams = DenseVector(newParams.toArray.dropRight(1))
    val newLogNoiseStdDev = newParams.toArray.last

    SgprModel(x, y, u, covFunc, newCovFuncParams, newLogNoiseStdDev)
  }
}