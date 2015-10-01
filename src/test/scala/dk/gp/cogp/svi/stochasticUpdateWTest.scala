package dk.gp.cogp.svi

import java.io.File
import org.junit._
import org.junit.Assert._
import breeze.linalg._
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.testutils.createCogpToyModel
import dk.gp.cogp.lb.calcLBLoglik
import dk.gp.cogp.testutils.loadToyModelData

class stochasticUpdateWTest {

  @Test def test_5_data_points = {

    val data = loadToyModelData(n = 5)

    val model = createCogpToyModel(data)

    val (newW, newWDelta) = stochasticUpdateW(LowerBound(model, data))

    val newModel = model.copy(w = newW)

    val loglik = calcLBLoglik(LowerBound(newModel, data))
    assertEquals(-121201.056728, loglik, 0.000001)

  }

  @Test def test_40_data_points = {

    val data = loadToyModelData(n = 40)

    val model = createCogpToyModel(data)

    val (newW, newWDelta) = stochasticUpdateW(LowerBound(model, data))

    val newModel = model.copy(w = newW)

    val loglik = calcLBLoglik(LowerBound(newModel, data))
    assertEquals(-9.210001189439e7, loglik, 0.0001)

  }

}