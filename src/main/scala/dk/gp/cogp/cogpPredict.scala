package dk.gp.cogp

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.inv
import breeze.numerics._
import breeze.linalg.cholesky
import dk.gp.math.invchol
import dk.gp.cogp.model.CogpModel
import breeze.linalg._
import dk.gp.cogp.util.calcKzzj
import dk.bayes.math.gaussian.Gaussian

//@TODO performance improvement, do not compute covariance  and inv(cov) for every single test point
object cogpPredict {

  /**
   * @param x [task index, prediction record]
   * @param model
   */
  def apply(x: DenseVector[Double], model: CogpModel): Gaussian = apply(x.toDenseMatrix, model)(0)

  /**
   * @param x [prediction record]
   * @param i task index
   * @param model
   */
  def apply(x: DenseVector[Double], i: Int, model: CogpModel): Gaussian = cogpPredict(x.toDenseMatrix, i, model)(0)

  /**
   * @param x Matrix of row vectors [prediction record]
   * @param i task index
   * @param model
   */
  def apply(x: DenseMatrix[Double], i: Int, model: CogpModel): DenseVector[Gaussian] = {
    val taskIndexVec = DenseMatrix.fill[Double](x.rows, 1)(i)
    val xTest = DenseMatrix.horzcat(taskIndexVec, x)

    apply(xTest, model)
  }

  /**
   * @param x Matrix of row vectors [task index, prediction record]
   * @param model
   */
  def apply(x: DenseMatrix[Double], model: CogpModel): DenseVector[Gaussian] = {

    val kZZj = calcKzzj(model)
    val w2 = pow(model.w, 2)

    x(*, ::).map { x =>
      val i = x(0).toInt
      val record = x(1 until x.size)
      pointPredict(record, i, model, kZZj, w2)
    }

  }

  def pointPredict(x: DenseVector[Double], i: Int, model: CogpModel, kZZj: Array[DenseMatrix[Double]], w2: DenseMatrix[Double]): Gaussian = {

    val w = model.w

    val predMj = predMean_j(x, model,kZZj)
    val predVarj = predVar_j(x, model, kZZj)

    val predMi = predMean_i(x, i, model)
    val predVari = predVar_i(x, i, model)

    val (outputM, outputVar) = if (model.g.size > 0) {
      val outputM = w(i, ::) * predMj + predMi
      val outputVar = w2(i, ::) * predVarj + predVari
      (outputM, outputVar)
    } else (predMi, predVari)

    Gaussian(outputM, outputVar)

  }

  private def predVar_i(s: DenseVector[Double], i: Int, model: CogpModel): Double = {

    val h = model.h(i)

    val kSS = h.covFunc.cov(s.toDenseMatrix, s.toDenseMatrix, h.covFuncParams)(0, 0)

    val kSZ = h.covFunc.cov(s.toDenseMatrix, h.z, h.covFuncParams)(0, ::)
    val kZZ = h.covFunc.cov(h.z, h.z, h.covFuncParams) + 1e-10 * DenseMatrix.eye[Double](h.z.rows)
    val kZZinv = invchol(cholesky(kZZ).t)

    val var_i = kSS - kSZ * (kZZinv - kZZinv * h.u.v * kZZinv) * kSZ.t
    var_i

  }

  private def predMean_i(s: DenseVector[Double], i: Int, model: CogpModel): Double = {

    val h = model.h(i)
    val kSZ = h.covFunc.cov(s.toDenseMatrix, h.z, h.covFuncParams)(0, ::)
    val kZZ = h.covFunc.cov(h.z, h.z, h.covFuncParams) + 1e-10 * DenseMatrix.eye[Double](h.z.rows)
    val kZZinv = invchol(cholesky(kZZ).t)
    val mean_i = kSZ * kZZinv * h.u.m
    mean_i

  }

  private def predMean_j(s: DenseVector[Double], model: CogpModel, kZZj: Array[DenseMatrix[Double]]): DenseVector[Double] = {

    val predMeanArray = model.g.zipWithIndex.map { case (g,j) =>

      val kSZ = g.covFunc.cov(s.toDenseMatrix, g.z, g.covFuncParams)(0, ::)
      val kZZ = kZZj(j)
      val kZZinv = invchol(cholesky(kZZ).t)

      kSZ * kZZinv * g.u.m
    }

    DenseVector(predMeanArray)
  }

  private def predVar_j(s: DenseVector[Double], model: CogpModel, kZZj: Array[DenseMatrix[Double]]) = {

    val predVarArray = model.g.zipWithIndex.map {
      case (g, j) =>

        val kSS = g.covFunc.cov(s.toDenseMatrix, s.toDenseMatrix, g.covFuncParams)(0, 0)

        val kSZ = g.covFunc.cov(s.toDenseMatrix, g.z, g.covFuncParams)(0, ::)
        val kZZ = kZZj(j)
        val kZZinv = invchol(cholesky(kZZ).t)

        kSS - kSZ * (kZZinv - kZZinv * g.u.v * kZZinv) * kSZ.t
    }

    DenseVector(predVarArray)
  }
}