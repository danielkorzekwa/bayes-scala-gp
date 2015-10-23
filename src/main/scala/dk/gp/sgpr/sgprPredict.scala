package dk.gp.sgpr

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.math.UnivariateGaussian
import breeze.linalg._
import breeze.numerics._

object sgprPredict {

  def apply(s: DenseMatrix[Double], model: SgprModel, computeVariance: Boolean = true): DenseVector[UnivariateGaussian] = {

    val likNoiseStdDev = exp(model.logNoiseStdDev)

    val predictedArray = (0 until s.rows).par.map { i =>

      val sRow = s(i, ::).t

      val kZU = model.covFunc.cov(sRow.toDenseMatrix, model.u, model.covFuncParams)
      val kZUinvLm = kZU * model.invLm
      val kZUinvLmInvLa = kZUinvLm * model.invLa

      val predMean = kZUinvLmInvLa * model.yKnmInvLmInvLa

      val predVariance = if (computeVariance) {

        val kZZ: DenseMatrix[Double] = model.covFunc.cov(sRow.toDenseMatrix, sRow.toDenseMatrix, model.covFuncParams)
        
        val v = diag(kZZ - kZUinvLm * kZUinvLm.t + pow(likNoiseStdDev, 2) * (kZUinvLmInvLa * kZUinvLmInvLa.t))
        v(0)
      } else Double.NaN
      UnivariateGaussian(predMean(0), predVariance)
    }.toArray

    DenseVector(predictedArray)

  }
}