package dk.gp.cogp.lb.grad

import org.junit._
import org.junit.Assert._
import breeze.linalg._
import java.io.File
import dk.gp.cogp.testutils.createCogpToyModel
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.testutils.loadToyModelData

class calcLBGradWTest {

  @Test def test_5_data_points = {

    val data = loadToyModelData(n = 5)

    val model = createCogpToyModel(data)

    val wGrad = calcLBGradW(LowerBound(model, data))
    assertEquals(-0.436041, wGrad(0, 0), 0.00001)
    assertEquals(-0.436041, wGrad(1, 0), 0.00001)
  }

  @Test def test_40_data_points = {

    val data = loadToyModelData(n = 40)

    val model = createCogpToyModel(data)

    val wGrad = calcLBGradW(LowerBound(model, data))
    assertEquals(-1.0137, wGrad(0, 0), 0.0001)
    assertEquals(-1.0137, wGrad(1, 0), 0.0001) //@TODO write a test, where  wGrad(0, 0)!=wGrad(1, 0)
  }
}