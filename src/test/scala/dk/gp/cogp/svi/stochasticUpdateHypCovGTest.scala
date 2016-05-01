package dk.gp.cogp.svi

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Test
import breeze.linalg.csvread
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.testutils.createCogpToyModel
import dk.gp.cogp.lb.calcLBLoglik
import dk.gp.cogp.testutils.loadToyModelData

class stochasticUpdateHypCovGTest {

  @Test def test_5_data_points = {

    val data = loadToyModelData(n = 5)

    val model = createCogpToyModel(data)

    val (newHypParams, newHypParamsDelta) = stochasticUpdateHypCovG(j = 0, LowerBound(model, data))

    val newG = model.g.head.copy(covFuncParams = newHypParams, covFuncParamsDelta = newHypParamsDelta)
    val newModel = model.copy(g = Array(newG))

    val loglik = calcLBLoglik(LowerBound(newModel, data))
    assertEquals(-209.55036, loglik, 0.00001)
  }

  @Test def test_40_data_points = {

    val data = loadToyModelData(n = 21)

    val model = createCogpToyModel(data)

    val (newHypParams, newHypParamsDelta) = stochasticUpdateHypCovG(j = 0, LowerBound(model, data))

    val newG = model.g.head.copy(covFuncParams = newHypParams, covFuncParamsDelta = newHypParamsDelta)
    val newModel = model.copy(g = Array(newG))

    val loglik = calcLBLoglik(LowerBound(newModel, data))
    assertEquals(-6660.2501, loglik, 0.0001)
  }

}