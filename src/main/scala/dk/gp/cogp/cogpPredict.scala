package dk.gp.cogp

import breeze.linalg.DenseMatrix
import dk.gp.math.UnivariateGaussian
import breeze.linalg.DenseVector
import breeze.linalg.inv
import breeze.numerics._
import breeze.linalg.cholesky
import dk.gp.math.invchol

object cogpPredict {

  def apply(sMatrix: DenseMatrix[Double], model: CogpModel): DenseMatrix[UnivariateGaussian] = {

    val predictedY = (0 until sMatrix.rows).map { sIndex =>
      val s = sMatrix(sIndex, ::).t
      cogpPredict(s, model)
    }

    DenseVector.horzcat[UnivariateGaussian](predictedY: _*).t
  }

  //@TODO it works with a single g function only
  def apply(s: DenseVector[Double], model: CogpModel): DenseVector[UnivariateGaussian] = {

    val u = model.g.map(_.u)
    val v = model.h.map(_.u)
    val w = model.w
    val beta = model.beta

    val kSS = model.g.head.covFunc.cov(s.toDenseMatrix.t, s.toDenseMatrix.t, model.g.head.covFuncParams)(0, 0)

    val kSZ = model.g.head.covFunc.cov(s.toDenseMatrix.t, model.g.head.z, model.g.head.covFuncParams)(0, ::)

    val kZZ = model.g.head.covFunc.cov(model.g.head.z, model.g.head.z, model.g.head.covFuncParams) + 1e-10 * DenseMatrix.eye[Double](model.g.head.z.size)
    val kZZinv = invchol(cholesky(kZZ).t)
    //val kZZinv = inv(kZZ)

    val w2 = pow(w, 2)

    val gM = DenseVector(u.map(u => kSZ * kZZinv * u.m))
    val gVar = DenseVector(u.map(u => kSS - kSZ * (kZZinv - kZZinv * u.v * kZZinv) * kSZ.t))

    val hM = v.map(v => kSZ * kZZinv * v.m)
    val hVar = DenseVector(v.map(v => kSS - kSZ * (kZZinv - kZZinv * v.v * kZZinv) * kSZ.t))

    val predictedOutputs = (0 until beta.size).map { i =>

      val outputM = w(i, ::) * gM + hM(i)
      val outputVar = w2(i, ::) * gVar + hVar(i)

      UnivariateGaussian(outputM, outputVar)

    }.toArray

    DenseVector(predictedOutputs)
  }
}