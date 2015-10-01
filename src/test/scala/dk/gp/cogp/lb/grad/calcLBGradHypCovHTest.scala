package dk.gp.cogp.lb.grad

import org.junit.Assert.assertEquals
import org.junit.Test

import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.testutils.createCogpToyModel
import dk.gp.cogp.testutils.loadToyModelData

class calcLBGradHypCovHTest {

  @Test def test_5_data_points = {

    val data = loadToyModelData(n = 5)

    val model = createCogpToyModel(data)

    val covParamsGrad0 = calcLBGradHypCovH(i = 0, LowerBound(model, data))
    assertEquals(-3.79732, covParamsGrad0(0), 0.00001)

    val covParamsGrad1 = calcLBGradHypCovH(i = 1, LowerBound(model,data))
    assertEquals(-4.01906, covParamsGrad1(0), 0.00001)

  }

  @Test def test_39_data_points = {

    val data = loadToyModelData(n = 40)

    val model = createCogpToyModel(data)

    val covParamsGrad0 = calcLBGradHypCovH(i = 0, LowerBound(model, data))
    assertEquals(48.40694, covParamsGrad0(0), 0.00001)

    val covParamsGrad1 = calcLBGradHypCovH(i = 1, LowerBound(model, data))
    assertEquals(47.79993, covParamsGrad1(0), 0.00001)

  }

}