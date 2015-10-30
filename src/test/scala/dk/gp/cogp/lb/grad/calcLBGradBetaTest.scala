package dk.gp.cogp.lb.grad

import org.junit.Assert.assertEquals
import org.junit.Test

import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.testutils.createCogpToyModel
import dk.gp.cogp.testutils.loadToyModelData

class calcLBGradBetaTest {

  @Test def test_5_data_points = {

    val data = loadToyModelData(n = 5)

    val model = createCogpToyModel(data)

    val betaGrad = calcLBGradBeta(LowerBound(model,data))
    assertEquals(-0.96381, betaGrad(0), 0.00001)
    assertEquals(-0.83919, betaGrad(1), 0.00001)

  }

  @Test def test_40_data_points = {

    val data = loadToyModelData(n = 40)

    val model = createCogpToyModel(data)

    val betaGrad = calcLBGradBeta(LowerBound(model,data))
    assertEquals(-52.24807, betaGrad(0), 0.00001)
    assertEquals(-51.90292, betaGrad(1), 0.00001)

  }
}