package dk.gp.sgpr

import dk.gp.cov.CovSEiso
import org.junit._
import Assert._
import breeze.linalg._
import java.io.File

class GenericSparseGPRTest {

  @Test def test_all_trainingset_as_inducing_points_not_large_scale = {
    val data = csvread(new File("src/test/resources/gpml/regression_data.csv"), skipLines = 1)
    val x = data(::, 0 to 0)
    val y = data(::, 1)

    val covFuncParams = DenseVector(0.68594, -0.99340)
    val covFunc = CovSEiso()
    val noiseLogStdDev = -1.9025

    val z = DenseVector(-1d, 1).toDenseMatrix.t

    val model = GenericSparseGPR(x, y, x, covFunc, covFuncParams, noiseLogStdDev)

    val predictions = model.predict(z)

    assertEquals(0.037, predictions(0).m, 0.0001) //z(0) mean
    assertEquals(0.01825, predictions(0).v, 0.0001) //z(0) variance

    assertEquals(0.9785, predictions(1).m, 0.0001) //z(1) mean
    assertEquals(1.7139, predictions(1).v, 0.0001) //z(1) variance
  }

  @Test def test_pseudo_inducing_points_not_large_scale = {
    val data = csvread(new File("src/test/resources/gpml/regression_data.csv"), skipLines = 1)
    val x = data(::, 0 to 0)
    val y = data(::, 1)

    val u = DenseVector.rangeD(-2, 2, 0.02).toDenseMatrix.t
    val covFuncParams = DenseVector(0.68594, -0.99340)
    val covFunc = CovSEiso()
    val noiseLogStdDev = -1.9025

    val z = DenseVector(-1d, 1).toDenseMatrix.t

    val model = GenericSparseGPR(x, y, u, covFunc, covFuncParams, noiseLogStdDev)

    val predictions = model.predict(z)

    assertEquals(0.0371, predictions(0).m, 0.0001) //z(0) mean
    assertEquals(0.01825, predictions(0).v, 0.0001) //z(0) variance

    assertEquals(0.9783, predictions(1).m, 0.0001) //z(1) mean
    assertEquals(1.7139, predictions(1).v, 0.0001) //z(1) variance
  }

  @Test def test_pseudo_inducing_points_large_scale = {

    val data = csvread(new File("src/test/resources/gpml/regression_data_4K.csv"), skipLines = 1)
    val x = data(::, 0 to 0)
    val y = data(::, 1)
    val u = DenseVector.rangeD(-20d, 20, 0.2).toDenseMatrix.t

    val covFuncParams = DenseVector(0.68594, -0.99340)
    val covFunc = CovSEiso()
    val noiseLogStdDev = -1.9025

    val z = DenseVector(-1d, 1).toDenseMatrix.t

    val model = GenericSparseGPR(x, y, u, covFunc, covFuncParams, noiseLogStdDev)

    val predictions = model.predict(z)

    assertEquals(0.5468, predictions(0).m, 0.0001) //z(0) mean
    assertEquals(8.3925e-4, predictions(0).v, 0.0001) //z(0) variance

    assertEquals(2.6868, predictions(1).m, 0.0001) //z(1) mean
    assertEquals(8.3923e-4, predictions(1).v, 0.0001) //z(1) variance
  }
}