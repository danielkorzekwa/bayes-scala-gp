package dk.gp.cogp.svi

import org.junit._
import org.junit.Assert._
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovSEiso
import scala.math._
import dk.gp.cov.CovSEiso
import breeze.linalg._
import java.io.File

class optimiseLBTest {

  val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))
  val x = data(::,0).toDenseMatrix.t
  val y = data(::,1 to 2)
 // val x: DenseMatrix[Double] = DenseVector.rangeD(-10, 10, 0.1).toDenseMatrix.t
 // val y = DenseVector.horzcat(DenseVector.zeros[Double](x.size) + 1d, DenseVector.zeros[Double](x.size) + 2d)

  val covFuncParams = DenseVector(log(1), log(1))
  val covFunc = CovSEiso()

  @Test def test = {

    val modelParams = optimiseLB(x, y, covFunc, covFuncParams, l = 0.1, iterNum = 100)

    println("w: " + modelParams.w)
    println("beta: " + modelParams.beta)
    println(modelParams.u(0).m)
    println(modelParams.v(0).m)
    println(modelParams.v(1).m)
  }
}