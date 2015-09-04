package dk.gp.cogp.svi.v

import java.io.File

import org.junit._
import org.junit.Assert._

import breeze.linalg._
import dk.gp.cogp.CogpModel
import dk.gp.cogp.calcLBLoglik
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.testutils.createCogpModel
import dk.gp.cov.CovFunc

class stochasticUpdateVTest {

  @Test def test_5_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpModel(x, y)

    val newH0 = model.h(0).copy(u = stochasticUpdateV(i = 0, LowerBound(model,x), model, x, y))
    val newH1 = model.h(1).copy(u = stochasticUpdateV(i = 1, LowerBound(model,x), model, x, y))

    val newModel = model.copy(h = Array(newH0, newH1))

    val loglik = calcLBLoglik(LowerBound(newModel, x), newModel, x, y)
    assertEquals(-121182.826893, loglik, 0.00001)

  }

  @Test def test_40_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 39, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpModel(x, y)

    val newH0 = model.h(0).copy(u = stochasticUpdateV(i = 0, LowerBound(model,x), model, x, y))
    val newH1 = model.h(1).copy(u = stochasticUpdateV(i = 1, LowerBound(model,x), model, x, y))

    val newModel = model.copy(h = Array(newH0, newH1))

    val loglik = calcLBLoglik(LowerBound(newModel, x), newModel, x, y)
    assertEquals(-9.2093955871883e7, loglik, 0.0001)

  }

}