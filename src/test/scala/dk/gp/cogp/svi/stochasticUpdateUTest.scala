package dk.gp.cogp.svi

import java.io.File
import org.junit._
import org.junit.Assert._
import breeze.linalg._
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.calcLBLoglik
import dk.gp.cogp.testutils.createCogpToyModel
import dk.gp.cogp.testutils.loadToyModelData

class stochasticUpdateUTest {

  @Test def test_5_data_points = {

    val data = loadToyModelData(n = 5)

    val model = createCogpToyModel(data)

    val newU = stochasticUpdateU(j = 0, LowerBound(model,data))

    val newG = model.g.head.copy(u = newU)
    val newModel = model.copy(g = Array(newG))

    val loglik = calcLBLoglik(LowerBound(newModel,data))
    assertEquals(-271.96521, loglik, 0.0001)
  }

  @Test def test_40_data_points = {

    val data = loadToyModelData(n = 40)

    val model = createCogpToyModel(data)

    val newU = stochasticUpdateU(j = 0,LowerBound(model,data))

    val newG = model.g.head.copy(u = newU)
    val newModel = model.copy(g = Array(newG))

    val loglik = calcLBLoglik(LowerBound(newModel,data))
    assertEquals(-11855.87165, loglik, 0.00001)
  }

}