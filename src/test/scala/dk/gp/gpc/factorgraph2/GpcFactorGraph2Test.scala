package dk.gp.gpc.factorgraph2

import java.io.File
import org.junit.Test
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.csvread
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.gp.gpc.TestCovFunc
import dk.bayes.factorgraph2.variable.CanonicalGaussianVariable
import dk.bayes.factorgraph2.variable._
import dk.bayes.factorgraph2.factor.CanonicalGaussianFactor
import dk.bayes.factorgraph2.factor.StepFunctionFactor
import dk.bayes.factorgraph2.api.calibrate
import dk.bayes.math.numericops.isIdentical
import dk.bayes.math.gaussian.canonical.CanonicalGaussian
import org.junit._
import Assert._

class GpcFactorGraph2Test {

  val x = csvread(new File("src/test/resources/gpml/classification_x.csv"))
  val y = csvread(new File("src/test/resources/gpml/classification_y.csv")).toDenseVector
  val t = csvread(new File("src/test/resources/gpml/classification_test.csv"))

  val covFunc = TestCovFunc()
  val covFuncParams = DenseVector(1.219809, 0.052334, 0.171199) //log sf, logEllx1, logEllx2
  val mean = -2.8499

  val covX = covFunc.cov(x, x, covFuncParams) + DenseMatrix.eye[Double](x.rows) * 1e-7
  val meanX = DenseVector.zeros[Double](x.rows) + mean

  @Test def test = {

    /**
     * create variables
     */

    val fVariable = CanonicalGaussianVariable()

    val yVariables = y.toArray.map { y =>
      val k = if (y == 1) 1 else 0
      BernVariable(k)
    }

    /**
     * Create factors
     */

    val fFactor = CanonicalGaussianFactor(fVariable, meanX, covX)

    val yFactors = y.toArray.zipWithIndex.map {
      case (y, i) =>
        StepFunctionFactor(fVariable, yVariables(i), x.rows, i, v = 1)
    }

    /**
     * Update variables
     */
    fVariable.update()

    /**
     * Calibration
     */
    var calibrated = false
    def calibrateStep() = {
      yFactors.foreach(yFactor => yFactor.updateMsgV1())

      val oldFMarginal = fVariable.get()

      fVariable.update()

      calibrated = CanonicalGaussian.isIdentical(oldFMarginal, fVariable.get, 1e-4)
    }

    val (calib, iter) = calibrate(calibrateStep, 100, calibrated)

    assertTrue(calib)
    assertEquals(44, iter)
    // println(fVariable.get.asInstanceOf[DenseCanonicalGaussian].mean)

    assertEquals(-1.411514, fVariable.get.asInstanceOf[DenseCanonicalGaussian].mean(0), 0.00001)
    assertEquals(-1.28964, fVariable.get.asInstanceOf[DenseCanonicalGaussian].mean(1), 0.00001)
    assertEquals(-6.3971, fVariable.get.asInstanceOf[DenseCanonicalGaussian].mean(2), 0.0001)
  }

}