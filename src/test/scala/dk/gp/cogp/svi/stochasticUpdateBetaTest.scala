package dk.gp.cogp.svi

import java.io.File
import org.junit._
import org.junit.Assert._
import breeze.linalg._
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.calcLBLoglik
import dk.gp.cogp.testutils.createCogpToyModel
import dk.gp.cogp.testutils.loadToyModelData

class stochasticUpdateBetaTest {

  @Test def test_5_data_points = {

    val data = loadToyModelData(n = 5)

    val model = createCogpToyModel(data)

    val (newBeta, newBetaDelta) = stochasticUpdateBeta(LowerBound(model, data))

    val newModel = model.copy(beta = newBeta)

    val loglik = calcLBLoglik(LowerBound(newModel, data))
    assertEquals(-121201.05671, loglik, 0.0001)
  }

  @Test def test_40_data_points = {

    val data = loadToyModelData(n = 40)

    val model = createCogpToyModel(data)

    val (newBeta, newBetaDelta) = stochasticUpdateBeta(LowerBound(model, data))

    val newModel = model.copy(beta = newBeta)

    val loglik = calcLBLoglik(LowerBound(newModel, data))
    assertEquals(-9.210001184018e7, loglik, 0.0001)
  }

}