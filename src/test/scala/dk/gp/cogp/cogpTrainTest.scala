package dk.gp.cogp

import org.junit.Test
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.numerics.log
import dk.gp.cov.CovSEiso
import breeze.linalg._
import java.io.File
import dk.gp.cov.CovSEiso
import dk.gp.cov.CovFunc
import dk.gp.cogp.svi.stochasticUpdateLB
import dk.gp.cogp.testutils.createCogpToyModel
import org.junit._
import Assert._
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.calcLBLoglik

class cogpTrainTest {

  @Test def test_5_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpToyModel(x, y)
    val newModel = cogpTrain(x, y, model, iterNum = 20)

    val loglik = calcLBLoglik(LowerBound(newModel, x), y)
    assertEquals(-109.61520, loglik, 0.00001)

  }

  @Test def test_40_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 39, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpToyModel(x, y)
    val newModel = cogpTrain(x, y, model, iterNum = 20)

    val loglik = calcLBLoglik(LowerBound(newModel, x), y)
    assertEquals(-1959.79424, loglik, 0.00001)

  }

  @Test def test = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 40, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val initialModel = createCogpToyModel(x, y)

    val finalModel = (1 to 50).foldLeft(initialModel) {
      case (currentModel, i) =>
        val newModel = stochasticUpdateLB(currentModel, x, y)

        val loglik = calcLBLoglik(LowerBound(newModel, x), y)
        println("LB loglik=" + loglik)

        newModel
    }

    val predictedY = cogpPredict(x, finalModel)

    //  println(predictedY)
  }
}