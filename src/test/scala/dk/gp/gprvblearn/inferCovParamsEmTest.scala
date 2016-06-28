package dk.gp.gprvblearn

import java.io.File

import scala.math._

import org.junit._
import org.junit.Assert._

import com.typesafe.scalalogging.slf4j.LazyLogging

import breeze.linalg._
import breeze.linalg.DenseMatrix
import dk.bayes.dsl.infer
import dk.bayes.dsl.variable.Gaussian
import dk.bayes.math.gaussian.MultivariateGaussian
import dk.gp.cov.CovSEiso
import dk.gp.gpr.GprModel
import dk.gp.gpr.gprLoglik

/**
 * Learning Gaussian Process covariance parameters by maximising variational lower bound:
 * The EM-EP Algorithm for Gaussian Process Classification
 *  (http://mlg.eng.cam.ac.uk/zoubin/papers/ecml03.pdf, http://mlg.eng.cam.ac.uk/zoubin/papers/KimGha06-PAMI.pdf)
 *
 *  @author Daniel Korzekwa
 */
class inferCovParamsEmTest extends LazyLogging {

  //[x,y]
  private val data = csvread(new File("src/test/resources/gpml/regression_data.csv"), skipLines = 1)
  private val x = data(::, 0 to 0)
  private val y = data(::, 1)
  val logLikStdDev = -1.9025

  @Test def test: Unit = {

    logger.info("Start learning GPR with EM-EP")

    //logSf, logEll
    val initParams = Array(log(1), log(1))

    val Array(finalLogSf, finalLogEll) = inferCovParamsEm(initParams, eStep, calcFPriorVar, calcFPriorVarD, maxIter = 200, tolerance = 1e-4)

    println("finalLogSf:" + finalLogSf)
    println("finalLogEll:" + finalLogEll)

    assertEquals(0.6879, finalLogSf, 0.01)
    assertEquals(-0.9902, finalLogEll, 0.01)
  }

  private def eStep(params: Array[Double]): MultivariateGaussian = {

    val m = DenseVector.zeros[Double](x.rows)
    val v = calcFPriorVar(params)
    val fVar = dk.bayes.dsl.variable.gaussian.multivariate.MultivariateGaussian(m, v)

    val likVarMatrix = pow(exp(logLikStdDev), 2) * DenseMatrix.eye[Double](x.rows)
    val yVar = Gaussian(fVar, likVarMatrix, yValue = y)

    val fPosterior = infer(fVar)

    //compute and print log likelihood for a given values of parameters
    val Array(logSf, logEll) = params
    val covFunc = CovSEiso()

    val gpModel = GprModel(x, y, covFunc, DenseVector(params), logLikStdDev)

      val meanX = gpModel.meanFunc(gpModel.x)
      val kXX = gpModel.calcKXX()
      val kXXInv = gpModel.calcKXXInv(kXX)
    
    val loglik = gprLoglik(meanX, kXX, kXXInv, gpModel.y)
    logger.info("Loglik=%.2f".format(loglik))

    MultivariateGaussian(fPosterior.m, fPosterior.v)
  }

  private def calcFPriorVar(params: Array[Double]): DenseMatrix[Double] = {
    val Array(logSf, logEll) = params
    val covFunc = CovSEiso()

    val v = covFunc.cov(x, x, DenseVector(params)) + DenseMatrix.eye[Double](x.rows) * 1e-10
    v
  }

  private def calcFPriorVarD(params: Array[Double]): Array[DenseMatrix[Double]] = {

    val Array(logSf, logEll) = params
    val covFunc = CovSEiso()

    val covD = covFunc.covD(x, x, DenseVector(params))
    covD
  }

}