package dk.gp.cogp.svi

import org.junit._
import Assert._
import breeze.linalg._
import dk.gp.cogp.testutils.createCogpToyModel
import java.io.File
import dk.gp.cogp.lb.calcLBLoglik
import dk.gp.cogp.lb.LowerBound

class stochasticUpdateCogpModelTest {

  @Test def test = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 40, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val initialModel = createCogpToyModel(x, y)

    val finalModel = (1 to 50).foldLeft(initialModel) {
      case (currentModel, i) =>
        val newModel = stochasticUpdateCogpModel(currentModel, x, y)

        val loglik = calcLBLoglik(LowerBound(newModel, x), y)
        // println("LB loglik=" + loglik)

        newModel
    }

    assertEquals(-1849.65675734641, calcLBLoglik(LowerBound(finalModel, x), y), 0.0001)

  }

}