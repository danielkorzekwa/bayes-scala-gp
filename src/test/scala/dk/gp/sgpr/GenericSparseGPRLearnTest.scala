package dk.gp.sgpr

import org.junit._
import Assert._
import breeze.linalg._
import java.io.File
import breeze.numerics._
import breeze.optimize.LBFGS

class GenericSparseGPRLearnTest {

  //Sample data
  //     val covFunc = CovSEiso(sf = 0.68594, ell = -0.99340)
  //      val noiseLogStdDev = -1.9025
  //    
  //       val x = Matrix((-20d to 20 by 0.01).toArray)
  //       println(x.numRows())
  //       val variance = covFunc.cov(x) + Matrix.identity(x.numRows())*pow(exp(noiseLogStdDev),2)
  //       val mvn = MultivariateGaussian(x,variance)
  //       val y = Matrix(mvn.draw(5656))
  //       x.toArray.zip(y.toArray).foreach(x => println(x._1 + "," + x._2))

  @Test def learn_pseudo_inducing_points_not_large_scale = {

    val data = csvread(new File("src/test/resources/gpml/regression_data.csv"), skipLines = 1)
    val x = data(::, 0 to 0)
    val y = data(::, 1)

    val u = DenseVector.rangeD(-2d,2,0.05).toDenseMatrix.t

    // logarithm of [signal standard deviation,length-scale,likelihood noise standard deviation] 
    val initialParams = DenseVector(log(1d), log(1), log(0.1))
    val diffFunction = SparseGpDiffFunction(x, y, u)

    val optimizer = new LBFGS[DenseVector[Double]](maxIter = 100, m = 3, tolerance = 1.0E-4)
    val optIterations = optimizer.iterations(diffFunction, initialParams).toList

    println("Initial loglik" + diffFunction.calculate(initialParams)._1)
    println("Final loglik" + diffFunction.calculate(optIterations.last.x)._1)
    assertEquals(13, optIterations.size)

    val newParams = optIterations.last.x
    //assert -negative log likelihood
    assertEquals(154.0760, diffFunction.calculate(initialParams)._1, 0.0001)
    assertEquals(14.1378, diffFunction.calculate(newParams)._1, 0.0001)

    assertEquals(0.7130, newParams(0), 0.0001)
    assertEquals(-0.99066, newParams(1), 0.0001)
    assertEquals(-1.8961, newParams(2), 0.0001)
    println("Learned gp parameters:" + newParams)
  }

  @Test def learn_pseudo_inducing_points_large_scale = {

    val data = csvread(new File("src/test/resources/gpml/regression_data_2K.csv"), skipLines = 1)
    val x = data(::, 0 to 0)
    val y = data(::, 1)

    val u = DenseVector.rangeD(-20d,0,1).toDenseMatrix.t

    // logarithm of [signal standard deviation,length-scale,likelihood noise standard deviation] 
    val initialParams = DenseVector(log(1d), log(1), log(0.1))
    val diffFunction = SparseGpDiffFunction(x, y, u)

    val optimizer = new LBFGS[DenseVector[Double]](maxIter = 100, m = 3, tolerance = 1.0E-4)
    val optIterations = optimizer.iterations(diffFunction, initialParams).toList

    println("Initial loglik" + diffFunction.calculate(initialParams)._1)
    println("Final loglik" + diffFunction.calculate(optIterations.last.x)._1)
    assertEquals(15, optIterations.size)

    val newParams = optIterations.last.x
    //assert -negative log likelihood
  //  assertEquals(68873.7356, diffFunction.calculate(initialParams)._1, 0.0001)
  //  assertEquals(2649.04376, diffFunction.calculate(newParams)._1, 0.0001)

    assertEquals(1.7715, newParams(0), 0.0001)
    assertEquals(0.3772, newParams(1), 0.0001)
    assertEquals(-0.0754, newParams(2), 0.0001)
    println("Learned gp parameters:" + newParams)
  }
}