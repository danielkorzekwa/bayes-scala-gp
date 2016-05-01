package dk.gp.cogp.model

import dk.gp.cov.CovFunc
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import dk.gp.math.invchol
import breeze.linalg.cholesky
import scala.util.Random
import breeze.stats._
import breeze.linalg.InjectNumericOps
import breeze.stats.variance.reduceDouble
import dk.bayes.math.gaussian.MultivariateGaussian

case class CogpGPVar(z: DenseMatrix[Double], u: MultivariateGaussian, covFunc: CovFunc, covFuncParams: DenseVector[Double], covFuncParamsDelta: DenseVector[Double]) {

  require(u.v.findAll (_.isNaN).size==0,"Inducing points variance is NaN:" + u.v)
  
  def calckZZ(): DenseMatrix[Double] = covFunc.cov(z, z, covFuncParams) + 1e-10 * DenseMatrix.eye[Double](z.rows)

  def calckXZ(x: DenseMatrix[Double]): DenseMatrix[Double] = covFunc.cov(x, z, covFuncParams)

  def calcdKzz(): Array[DenseMatrix[Double]] = covFunc.covD(z, z, covFuncParams)

  def calcdKxz(x: DenseMatrix[Double]): Array[DenseMatrix[Double]] = covFunc.covD(x, z, covFuncParams)

}

object CogpGPVar {

  def apply(y: DenseVector[Double], z: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double]): CogpGPVar = {

    CogpGPVar(z, getInitialIndVar(y, z), covFunc, covFuncParams, DenseVector.zeros[Double](covFuncParams.size))
  }

  private def getInitialIndVar(y: DenseVector[Double], z: DenseMatrix[Double]): MultivariateGaussian = {

    val m = DenseVector.zeros[Double](z.rows)    
    val v = 10*variance(y)*DenseMatrix.eye[Double](z.rows)
    
    MultivariateGaussian(m, v)
  }
}