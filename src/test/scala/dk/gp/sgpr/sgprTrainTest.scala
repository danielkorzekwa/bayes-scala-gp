package dk.gp.sgpr

import org.junit._
import Assert._
import breeze.linalg._
import breeze.numerics._
import java.io.File
import dk.gp.cov.CovSEiso

class sgprTrainTest {

  @Test def learn_pseudo_inducing_points_not_large_scale = {
    val data = csvread(new File("src/test/resources/gpml/regression_data.csv"), skipLines = 1)
    val x = data(::, 0 to 0)
    val y = data(::, 1)

    val u = DenseVector.rangeD(-2d, 2, 0.05).toDenseMatrix.t

    val covFunc = CovSEiso()
    // logarithm of [signal standard deviation,length-scale] 
    val initialCovParams = DenseVector(log(1d), log(1))
    val logNoiseStdDev = log(0.1)

    val sgprModel = sgprTrain(x, y, u, covFunc, initialCovParams, logNoiseStdDev)

    assertEquals(0.7130, sgprModel.covFuncParams(0), 0.0001)
    assertEquals(-0.99066, sgprModel.covFuncParams(1), 0.0001)
    assertEquals(-1.8961, sgprModel.logNoiseStdDev, 0.0001)
  }

  @Test def learn_pseudo_inducing_points_large_scale = {
    val data = csvread(new File("src/test/resources/gpml/regression_data_2K.csv"), skipLines = 1)
    val x = data(::, 0 to 0)
    val y = data(::, 1)

    val u = DenseVector.rangeD(-20d, 0, 1).toDenseMatrix.t

    val covFunc = CovSEiso()
    // logarithm of [signal standard deviation,length-scale] 
    val initialCovParams = DenseVector(log(1d), log(1))
    val logNoiseStdDev = log(0.1)

    val sgprModel = sgprTrain(x, y, u, covFunc, initialCovParams, logNoiseStdDev)

    assertEquals(1.7715, sgprModel.covFuncParams(0), 0.0001)
    assertEquals(0.3772, sgprModel.covFuncParams(1), 0.0001)
    assertEquals(-0.0754, sgprModel.logNoiseStdDev, 0.0001)
  }
}