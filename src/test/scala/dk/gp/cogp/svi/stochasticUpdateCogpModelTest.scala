package dk.gp.cogp.svi

import org.junit._
import Assert._
import breeze.linalg._
import dk.gp.cogp.testutils.createCogpToyModel
import java.io.File
import dk.gp.cogp.lb.calcLBLoglik
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.testutils.loadToyModelData

class stochasticUpdateCogpModelTest {

  @Test def test = {

    val data = loadToyModelData(n = 41)

    val initialModel = createCogpToyModel(data)

    val finalModel = (1 to 50).foldLeft(initialModel) {
      case (currentModel, i) =>
        val newModel = stochasticUpdateCogpModel(LowerBound(currentModel, data),data).model

        val loglik = calcLBLoglik(LowerBound(newModel, data))
        // println("LB loglik=" + loglik)

        newModel
    }

    assertEquals(-1849.654, calcLBLoglik(LowerBound(finalModel, data)), 0.001)

  }

}