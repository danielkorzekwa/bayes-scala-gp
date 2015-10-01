package dk.gp.cogp.svi

import java.io.File

import org.junit._
import org.junit.Assert._

import breeze.linalg._
import dk.gp.cogp.lb._
import dk.gp.cogp.testutils.createCogpToyModel
import dk.gp.cov.CovFunc
import dk.gp.cogp.testutils._

class stochasticUpdateVTest {

  @Test def test_5_data_points = {

    val data = loadToyModelData(n = 5)

    val model = createCogpToyModel(data)

    val newH0 = model.h(0).copy(u = stochasticUpdateV(i = 0, LowerBound(model, data)))
    val newH1 = model.h(1).copy(u = stochasticUpdateV(i = 1, LowerBound(model, data)))

    val newModel = model.copy(h = Array(newH0, newH1))

    val loglik = calcLBLoglik(LowerBound(newModel, data))
    assertEquals(-121182.692052, loglik, 0.00001)

  }

  @Test def test_40_data_points = {

    val data = loadToyModelData(n = 40)

    val model = createCogpToyModel(data)

    val newH0 = model.h(0).copy(u = stochasticUpdateV(i = 0, LowerBound(model, data)))
    val newH1 = model.h(1).copy(u = stochasticUpdateV(i = 1, LowerBound(model, data)))

    val newModel = model.copy(h = Array(newH0, newH1))

    val loglik = calcLBLoglik(LowerBound(newModel, data))
    assertEquals(-9.209392827281e7, loglik, 0.0001)

  }

}