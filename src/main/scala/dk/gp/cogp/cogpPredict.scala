package dk.gp.cogp

import breeze.linalg.DenseMatrix
import dk.gp.math.UnivariateGaussian
import breeze.linalg.DenseVector
import breeze.linalg.inv
import breeze.numerics._
import breeze.linalg.cholesky
import dk.gp.math.invchol
import dk.gp.cogp.model.CogpModel
import breeze.linalg._

//@TODO performance improvement, do not compute covariance  and inv(cov) for every single test point
object cogpPredict {

  def apply(x: DenseVector[Double], model: CogpModel): DenseMatrix[UnivariateGaussian] = {
    cogpPredict(x.toDenseMatrix.t, model)
  }

  def apply(x: DenseMatrix[Double], model: CogpModel): DenseMatrix[UnivariateGaussian] = x(*, ::).map(r => pointPredict(r, model))

  def pointPredict(x: DenseVector[Double], model: CogpModel): DenseVector[UnivariateGaussian] = {

    val w = model.w
    val w2 = pow(w, 2)

    val predMj = predMean_j(x, model)
    val predVarj = predVar_j(x, model)

    val predMi = predMean_i(x, model)
    val predVari = predVar_i(x, model)

    val predictedOutputs = (0 until model.h.size).map { i =>

      val (outputM, outputVar) = if (model.g.size > 0) {
        val outputM = w(i, ::) * predMj + predMi(i)
        val outputVar = w2(i, ::) * predVarj + predVari(i)
        (outputM, outputVar)
      } else (predMi(i), predVari(i))

      UnivariateGaussian(outputM, outputVar)

    }.toArray

    DenseVector(predictedOutputs)
  }

  private def predVar_i(s: DenseVector[Double], model: CogpModel): Array[Double] = {

    val predVarArray = model.h.map { h =>

      val kSS = h.covFunc.cov(s.toDenseMatrix.t, s.toDenseMatrix.t, h.covFuncParams)(0, 0)

      val kSZ = h.covFunc.cov(s.toDenseMatrix.t, h.z, h.covFuncParams)(0, ::)
      val kZZ = h.covFunc.cov(h.z, h.z, h.covFuncParams) + 1e-10 * DenseMatrix.eye[Double](h.z.size)
      val kZZinv = invchol(cholesky(kZZ).t)

      kSS - kSZ * (kZZinv - kZZinv * h.u.v * kZZinv) * kSZ.t

    }

    predVarArray
  }

  private def predMean_i(s: DenseVector[Double], model: CogpModel): Array[Double] = {

    val predMeanArray = model.h.map { h =>

      val kSZ = h.covFunc.cov(s.toDenseMatrix.t, h.z, h.covFuncParams)(0, ::)
      val kZZ = h.covFunc.cov(h.z, h.z, h.covFuncParams) + 1e-10 * DenseMatrix.eye[Double](h.z.size)
      val kZZinv = invchol(cholesky(kZZ).t)
      kSZ * kZZinv * h.u.m
    }

    predMeanArray

  }

  private def predMean_j(s: DenseVector[Double], model: CogpModel): DenseVector[Double] = {

    val predMeanArray = model.g.map { g =>

      val kSZ = g.covFunc.cov(s.toDenseMatrix.t, g.z, g.covFuncParams)(0, ::)
      val kZZ = g.covFunc.cov(g.z, g.z, g.covFuncParams) + 1e-10 * DenseMatrix.eye[Double](g.z.size)
      val kZZinv = invchol(cholesky(kZZ).t)

      kSZ * kZZinv * g.u.m
    }

    DenseVector(predMeanArray)
  }

  private def predVar_j(s: DenseVector[Double], model: CogpModel) = {

    val predVarArray = model.g.map { g =>

      val kSS = g.covFunc.cov(s.toDenseMatrix.t, s.toDenseMatrix.t, g.covFuncParams)(0, 0)

      val kSZ = g.covFunc.cov(s.toDenseMatrix.t, g.z, g.covFuncParams)(0, ::)
      val kZZ = g.covFunc.cov(g.z, g.z, g.covFuncParams) + 1e-10 * DenseMatrix.eye[Double](g.z.size) //@TODO it is also computed in LowerBound, use a single implementation in one place, e.g. in CogpGPVar, in many places in cogp project
      val kZZinv = invchol(cholesky(kZZ).t)

      kSS - kSZ * (kZZinv - kZZinv * g.u.v * kZZinv) * kSZ.t
    }

    DenseVector(predVarArray)
  }
}