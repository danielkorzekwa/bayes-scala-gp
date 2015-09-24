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
import dk.gp.cogp.testutils.createCogpToyModel
import org.junit._
import Assert._
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.calcLBLoglik
import dk.gp.cogp.svi.stochasticUpdateCogpModel
import scala.util.Random
import dk.gp.cogp.testutils.loadToyModelData
import dk.gp.cogp.testutils.createSingleHZeroGModel
import dk.gp.cogp.testutils.createOneHOneGModel
import dk.gp.cogp.testutils.createOneHCovSEisoOneCovNoiseGModel

class cogpTrainTest {

  @Test def test_5_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpToyModel(x, y)
    val newModel = cogpTrain(x, y, model, iterNum = 20)

    val loglik = calcLBLoglik(LowerBound(newModel, x, y))
    assertEquals(-109.6151, loglik, 0.0001)

  }

  @Test def test_40_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 39, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpToyModel(x, y)
    val newModel = cogpTrain(x, y, model, iterNum = 20)

    val loglik = calcLBLoglik(LowerBound(newModel, x, y))
    assertEquals(-1959.8154, loglik, 0.0001)

    val predictedY = cogpPredict(x, model)

    assertEquals(0, predictedY(10, 0).m, 0.0001)

  }

  @Test def test_40_data_points_500_iter = {

    Random.setSeed(4676)

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 39, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)
    val z = x(0 until x.rows by 10, ::) // inducing points for u and v inducing variables
    val model = createCogpToyModel(x, y, z)
    val newModel = cogpTrain(x, y, model, iterNum = 500)

    val loglik = calcLBLoglik(LowerBound(newModel, x, y))
    assertEquals(76.4803, loglik, 0.0001)

    val predictedY = cogpPredict(x, newModel)

    assertEquals(-0.4271, predictedY(10, 0).m, 0.0001)
    assertEquals(0.00257, predictedY(10, 0).v, 0.0001)

    assertEquals(0.4260, predictedY(10, 1).m, 0.0001)
    assertEquals(0.00257, predictedY(10, 1).v, 0.0001)

  }

  @Test def test_with_inducing_points_500_iter_missing_points = {

    Random.setSeed(4676)

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 39, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    y(5 to 10, 0) := Double.NaN
    y(30 to 35, 1) := Double.NaN

    val z = x(0 until x.rows by 10, ::) // inducing points for u and v inducing variables
    val model = createCogpToyModel(x, y, z)
    val newModel = cogpTrain(x, y, model, iterNum = 500)

    val loglik = calcLBLoglik(LowerBound(newModel, x, y))
    assertEquals(60.1350, loglik, 0.0001)

    val predictedY = cogpPredict(x, newModel)

    assertEquals(-0.4279, predictedY(10, 0).m, 0.0001)
    assertEquals(0.00378, predictedY(10, 0).v, 0.0001)

    assertEquals(0.4276, predictedY(10, 1).m, 0.0001)
    assertEquals(0.00294, predictedY(10, 1).v, 0.0001)

  }

  @Test def test_single_h_zero_g = {

    Random.setSeed(4676)
    val data = loadToyModelData()
    val x = data(::, 0)
    val y = data(::, 1 to 1)

    val z = x(0 until x.size by 10) // inducing points for u and v inducing variables
    val model = createSingleHZeroGModel(x, y, z)
    val newModel = cogpTrain(x, y, model, iterNum = 200)

    val loglik = calcLBLoglik(LowerBound(newModel, x, y))
    assertEquals(207.6311, loglik, 0.0001)

    val predictedY = cogpPredict(x, newModel)

    assertEquals(-0.41389, predictedY(10, 0).m, 0.0001)
    assertEquals(0.001100, predictedY(10, 0).v, 0.0001)

  }

  @Test def test_single_h_single_g = {

    Random.setSeed(4676)
    val data = loadToyModelData()
    val x = data(::, 0)
    val y = data(::, 1 to 1)

    val z = x(0 until x.size by 10) // inducing points for u and v inducing variables
    val model = createOneHOneGModel(x, y, z)
    val newModel = cogpTrain(x, y, model, iterNum = 200)

    val loglik = calcLBLoglik(LowerBound(newModel, x, y))
    assertEquals(191.8406, loglik, 0.0001)

    val predictedY = cogpPredict(x, newModel)

    assertEquals(-0.4127, predictedY(10, 0).m, 0.0001)
    assertEquals(0.0016, predictedY(10, 0).v, 0.0001)

  }

  @Test def test_40_data_points_500_iter_single_h_covseiso_single_g_cov_noise = {

    val data = loadToyModelData()(0 to 39, ::)
    val x = data(::, 0)
    val y = data(::, 1 to 1)

    val z = x
    val model = createOneHCovSEisoOneCovNoiseGModel(x, y, z)
    val newModel = cogpTrain(x, y, model, iterNum = 200)

    val loglik = calcLBLoglik(LowerBound(newModel, x, y))
    assertEquals(-42.264, loglik, 0.01)

    val predictedY = cogpPredict(x, newModel)

    assertEquals(-0.39595, predictedY(10, 0).m, 0.0001)
    assertEquals(0.00579, predictedY(10, 0).v, 0.0001)

  }

}