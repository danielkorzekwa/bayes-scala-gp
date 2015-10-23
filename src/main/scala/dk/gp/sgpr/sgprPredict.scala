package dk.gp.sgpr

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.math.UnivariateGaussian
import breeze.linalg._
import breeze.numerics._

object sgprPredict {

  def apply(s: DenseMatrix[Double], model: SgprModel, computeVariance: Boolean = true): DenseVector[UnivariateGaussian] = {

    val likNoiseStdDev = exp(model.logNoiseStdDev)
    val invLmInvLa = model.invLm * model.invLa
    val invLminvLayKnmInvLmInvLa = (invLmInvLa * model.yKnmInvLmInvLa)

    val predictedArray = (0 until s.rows).par.map { i =>

      val sRow = s(i, ::).t
      val kZU = model.covFunc.cov(sRow.toDenseMatrix, model.u, model.covFuncParams)

      val predMean = kZU * invLminvLayKnmInvLmInvLa
      val predVariance = if (computeVariance) {

        val kZZ: DenseMatrix[Double] = model.covFunc.cov(sRow.toDenseMatrix, sRow.toDenseMatrix, model.covFuncParams)

        val kZUinvLmInvLa = kZU * invLmInvLa
        val kZUinvLm = kZU * model.invLm

        val v = diag(kZZ - kZUinvLm * kZUinvLm.t + pow(likNoiseStdDev, 2) * (kZUinvLmInvLa * kZUinvLmInvLa.t))
        v(0)
      } else Double.NaN
      UnivariateGaussian(predMean(0), predVariance)
    }.toArray

    DenseVector(predictedArray)

  }
}