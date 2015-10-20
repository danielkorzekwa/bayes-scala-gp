package dk.gp.sgpr

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.math.UnivariateGaussian
import breeze.linalg._
import breeze.numerics._

object sgprPredict {

  def apply(s: DenseMatrix[Double], model: SgprModel): DenseVector[UnivariateGaussian] = {

    val likNoiseStdDev = exp(model.logNoiseStdDev)
    
    val predicted = s(*, ::).map { s =>

      val kZZ: DenseMatrix[Double] = model.covFunc.cov(s.toDenseMatrix, s.toDenseMatrix, model.covFuncParams)
      val kZU: DenseMatrix[Double] = model.covFunc.cov(s.toDenseMatrix, model.u, model.covFuncParams)
      val kUZ = kZU.t

      val predMean = pow(likNoiseStdDev, -2) * kZU * model.sigmaKmnyVal
      val predVariance = kZZ - kZU * model.kMMinv * kUZ + kZU * model.sigma * kUZ
      UnivariateGaussian(predMean(0), predVariance(0, 0))
    }

    predicted
  }
}