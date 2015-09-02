package dk.gp.cogp.svi.hypcovg

import org.junit._
import Assert._
import breeze.linalg._
import java.io.File
import dk.gp.cov.CovFunc
import dk.gp.cov.CovSEiso
import breeze.numerics.log
import dk.gp.cogp.CogpModel
import dk.gp.cogp.testutils.createCogpModel

class calcLBGradHypCovGTest {

  @Test def test_5_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpModel(x, y)

    val covParamsGrad = calcLBGradHypCovG(j = 0, model, x, y)
    assertEquals(235416.94955, covParamsGrad(0), 0.00001)
    assertEquals(-939897.979837, covParamsGrad(1), 0.00001)
  }

  @Test def test_40_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 39, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpModel(x, y)

    val covParamsGrad = calcLBGradHypCovG(j = 0, model, x, y)
    assertEquals(2647306.86845, covParamsGrad(0), 0.00001)
    assertEquals(-3.193795078027e7, covParamsGrad(1), 0.00001)
  }

}