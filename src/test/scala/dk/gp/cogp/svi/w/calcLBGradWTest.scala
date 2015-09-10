package dk.gp.cogp.svi.w

import org.junit._
import Assert._
import breeze.linalg._
import java.io.File
import dk.gp.cov.CovFunc
import dk.gp.cov.CovSEiso
import breeze.numerics.log
import dk.gp.cogp.CogpModel
import dk.gp.cogp.testutils.createCogpToyModel
import dk.gp.cogp.lb.LowerBound

class calcLBGradWTest {

  @Test def test_5_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpToyModel(x, y)

    val wGrad = calcLBGradW(LowerBound(model,x), y)
    assertEquals(-0.436026, wGrad(0, 0), 0.00001)
    assertEquals(-0.436026, wGrad(1, 0), 0.00001)
  }

  @Test def test_40_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 39, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpToyModel(x, y)

    val wGrad = calcLBGradW(LowerBound(model,x), y)
    assertEquals(-1.014200, wGrad(0, 0), 0.00001)
    assertEquals(-1.014200, wGrad(1, 0), 0.00001) //@TODO write a test, where  wGrad(0, 0)!=wGrad(1, 0)
  }
}