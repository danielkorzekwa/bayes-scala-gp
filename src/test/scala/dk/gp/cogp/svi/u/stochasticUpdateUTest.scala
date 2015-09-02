package dk.gp.cogp.svi.u

import org.junit._
import Assert._
import breeze.linalg._
import java.io.File
import dk.gp.cov.CovFunc
import dk.gp.cov.CovSEiso
import breeze.numerics.log
import dk.gp.cogp.CogpModel
import dk.gp.cogp.calcLBLoglik
import dk.gp.cogp.testutils.createCogpModel

class stochasticUpdateUTest {

  @Test def test_5_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpModel(x, y)

    val newU = stochasticUpdateU(j = 0, model, x, y)

    val newG = model.g.head.copy(u = newU)
    val newModel = model.copy(g = Array(newG))

    val loglik = calcLBLoglik(newModel, x, y)
    assertEquals(-271.96523, loglik, 0.00001)
  }

  @Test def test_40_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 39, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpModel(x, y)

    val newU = stochasticUpdateU(j = 0, model, x, y)

    val newG = model.g.head.copy(u = newU)
    val newModel = model.copy(g = Array(newG))

    val loglik = calcLBLoglik(newModel, x, y)
    assertEquals(-11855.84340, loglik, 0.00001)
  }

}