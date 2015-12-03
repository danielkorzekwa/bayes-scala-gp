package dk.gp.gpc.factorgraph2

import java.io.File

import org.junit.Test

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.csvread
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.gp.gpc.TestCovFunc

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

    val fVariable = GaussianVariable()

    val yVariables = y.toArray.map(y => BernVariable(y))

    /**
     * Create factors
     */

    val fFactor = MultivariateGaussianFactor(fVariable, meanX, covX)

    val yFactors = y.toArray.zipWithIndex.map {
      case (y, i) =>
        MvnGaussianThresholdFactor(fVariable, yVariables(i), x.rows, i, v = 1)
    }

    /**
     * Update variables
     */
    fVariable.update()
    yVariables.foreach(_.update())

    /**
     * Calibration
     */

    def calibrateStep() = {
      yFactors.foreach(yFactor => yFactor.updateMsgV1())
      fVariable.update()
    }

    for (i <- 0 until 10) {
      calibrateStep()
    }

    println(fVariable.get.asInstanceOf[DenseCanonicalGaussian].mean)

  }
}