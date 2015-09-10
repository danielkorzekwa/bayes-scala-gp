package dk.gp.cogp.lb.grad

import org.junit._
import org.junit.Assert._
import breeze.linalg._
import java.io.File
import dk.gp.cogp.testutils.createCogpToyModel
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.grad.calcLBGradHypCovG

class calcLBGradHypCovGTest {

  @Test def test_5_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpToyModel(x, y)

    val covParamsGrad = calcLBGradHypCovG(j = 0, LowerBound(model, x), y)
    assertEquals(235416.68923, covParamsGrad(0), 0.00001)
    assertEquals(-939896.15629, covParamsGrad(1), 0.00001)
  }

  @Test def test_40_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 39, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpToyModel(x, y)

    val covParamsGrad = calcLBGradHypCovG(j = 0, LowerBound(model, x), y)
    assertEquals(2647496.162073, covParamsGrad(0), 0.00001)
    assertEquals(-3.19382182328e7, covParamsGrad(1), 0.00001)
  }

}