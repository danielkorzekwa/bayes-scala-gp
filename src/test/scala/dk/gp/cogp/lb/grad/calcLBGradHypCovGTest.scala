package dk.gp.cogp.lb.grad

import org.junit.Assert.assertEquals
import org.junit.Test

import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.testutils.createCogpToyModel
import dk.gp.cogp.testutils.loadToyModelData

class calcLBGradHypCovGTest {

  @Test def test_5_data_points = {

    val data = loadToyModelData(n = 5)

    val model = createCogpToyModel(data)

    val covParamsGrad = calcLBGradHypCovG(j = 0, LowerBound(model,data))
    assertEquals(235416.68923, covParamsGrad(0), 0.00001)
    assertEquals(-939896.15629, covParamsGrad(1), 0.00001)
  }

  @Test def test_40_data_points = {

    val data = loadToyModelData(n = 40)

    val model = createCogpToyModel(data)

    val covParamsGrad = calcLBGradHypCovG(j = 0, LowerBound(model, data))
    assertEquals(2647496.162073, covParamsGrad(0), 0.00001)
    assertEquals(-3.19382182328e7, covParamsGrad(1), 0.00001)
  }

}