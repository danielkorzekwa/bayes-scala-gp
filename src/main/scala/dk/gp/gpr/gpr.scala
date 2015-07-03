package dk.gp.gpr

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import breeze.optimize.LBFGS
import breeze.linalg.VectorConstructors

object gpr {

  def apply(x: DenseMatrix[Double], y: DenseVector[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], noiseLogStdDev: Double, mean: Double = 0d,gpMaxIter:Int=100): GprModel = {

    val initialGpModel = GprModel(x, y, covFunc, covFuncParams, noiseLogStdDev, mean)
    
    val diffFunc = GprDiffFunction(initialGpModel)
    val initialParams = DenseVector(covFuncParams.toArray :+ noiseLogStdDev)
    
    val optimizer = new LBFGS[DenseVector[Double]](maxIter = gpMaxIter, m = 3, tolerance = 1.0E-6)
    val optIterations = optimizer.iterations(diffFunc, initialParams).toList

    val newParams = optIterations.last.x

    val newCovFuncParams = DenseVector(newParams.toArray.dropRight(1))
    val newNoiseLogStdDev = newParams.toArray.last

    val trainedGpModel = GprModel(x,y,covFunc,newCovFuncParams,newNoiseLogStdDev,mean)
    trainedGpModel
  }
}

