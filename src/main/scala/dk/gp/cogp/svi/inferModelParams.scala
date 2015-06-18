package dk.gp.cogp.svi

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cogp.model.CogpModelParams
import dk.gp.cov.CovFunc
import dk.gp.math.MultivariateGaussian

object inferModelParams {

  def apply(x: DenseMatrix[Double], y: DenseMatrix[Double], covFunc: CovFunc, l: Double, iterNum: Int): CogpModelParams = {

    //likelihood noise precision
    val beta = DenseVector(1d / 0.01, 1d / 0.02) // [P x 1]

    //mixing weights
    val w = new DenseMatrix(2, 1, Array(1.1, -1.2)) // [P x Q]

    val z = x //simplifying assumption
    val kZZ = covFunc.covNM(x, z)

    val priorU = Array(MultivariateGaussian(DenseVector.zeros[Double](x.size), kZZ))
    val priorV = Array(MultivariateGaussian(DenseVector.zeros[Double](x.size), kZZ), MultivariateGaussian(DenseVector.zeros[Double](x.size), kZZ))

    val priorModelParams = CogpModelParams(priorU, priorV, beta, w)

   val learnedModelParams = (0 until iterNum).foldLeft(priorModelParams) { case (currModelParams, iter) => stochasticUpdateModelParams(currModelParams, x, y, covFunc, l) }
   learnedModelParams
  }
}