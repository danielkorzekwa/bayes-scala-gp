package dk.gp.cogp.svi.hypcovh

import java.io.File
import org.junit._
import org.junit.Assert._
import breeze.linalg._
import dk.gp.cogp.CogpModel
import dk.gp.cogp.testutils.createCogpModel
import dk.gp.cogp.lb.LowerBound

class calcLBGradHypCovHTest {

  @Test def test_5_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpModel(x, y)

    val covParamsGrad0 = calcLBGradHypCovH(i = 0, LowerBound(model, x),  y)
    assertEquals(-3.79732, covParamsGrad0(0), 0.00001)

    val covParamsGrad1 = calcLBGradHypCovH(i = 1, LowerBound(model, x),  y)
    assertEquals(-4.01906, covParamsGrad1(0), 0.00001)

  }

  @Test def test_39_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 39, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpModel(x, y)

    val covParamsGrad0 = calcLBGradHypCovH(i = 0, LowerBound(model, x),  y)
    assertEquals(48.40694, covParamsGrad0(0), 0.00001)

    val covParamsGrad1 = calcLBGradHypCovH(i = 1, LowerBound(model, x),  y)
    assertEquals(47.79993, covParamsGrad1(0), 0.00001)

  }

}