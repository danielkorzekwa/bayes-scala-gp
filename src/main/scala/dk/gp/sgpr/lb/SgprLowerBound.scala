package dk.gp.sgpr.lb

import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import breeze.numerics._
import breeze.linalg.DenseMatrix
import dk.gp.math.invchol
import breeze.linalg.cholesky
import breeze.linalg.inv
import breeze.stats._
import dk.gp.cov.utils.covDiag
import dk.gp.cov.utils.covDiagD
import breeze.linalg.InjectNumericOps
case class SgprLowerBound(x: DenseMatrix[Double], y: DenseVector[Double], u: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], logNoiseStdDev: Double) {

  val likNoiseStdDev = exp(logNoiseStdDev)
  val likNoiseVar = likNoiseStdDev * likNoiseStdDev

  val kMM: DenseMatrix[Double] = covFunc.cov(u, u, covFuncParams)
  val lm: DenseMatrix[Double] = cholesky(kMM + 1e-7 * DenseMatrix.eye[Double](kMM.rows)).t
  val kMMinv = invchol(lm)

  val kMN: DenseMatrix[Double] = covFunc.cov(u, x, covFuncParams)
  val kNM = kMN.t

  val kNNdiag = covDiag(x, covFunc, covFuncParams)

  val kNNDiagDArray = covDiagD(x, covFunc, covFuncParams)
  val kMMdArray = covFunc.covD(u, u, covFuncParams)
  val kNMdArray = covFunc.covD(x, u, covFuncParams)

  val kMNkNM = kMN * kNM

  val yy = y.t * y

  val invLm = inv(lm) 
  val KnmInvLm = kNM * invLm
  val C = KnmInvLm.t * KnmInvLm
  val a = likNoiseVar * DenseMatrix.eye[Double](u.rows) + C
  val la = cholesky(a).t
  val invLa = inv(la)

  val aInv = (invLm * invLa) * (invLm * invLa).t
  val aInvkMNy = aInv * kMN * y
  
   val yKnmInvLmInvLa = (y.t * kNM * invLm) * invLa

}