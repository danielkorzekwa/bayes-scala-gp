package dk.gp.cogp.svi.beta

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

class calcLBGradBetaTest {

  @Test def test_5_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpToyModel(x, y)

    val betaGrad = calcLBGradBeta(LowerBound(model,x), y)
    assertEquals(-0.96381, betaGrad(0), 0.00001)
    assertEquals(-0.83919, betaGrad(1), 0.00001)

  }

  @Test def test_40_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 39, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpToyModel(x, y)

    val betaGrad = calcLBGradBeta(LowerBound(model,x), y)
    assertEquals(-52.24807, betaGrad(0), 0.00001)
    assertEquals(-51.90292, betaGrad(1), 0.00001)

  }
}