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
import dk.gp.cov.CovFunc

class optimiseLBTest {

  val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))
  val x = data(::, 0).toDenseMatrix.t
  val y = data(::, 1 to 2)
  // val x: DenseMatrix[Double] = DenseVector.rangeD(-10, 10, 0.1).toDenseMatrix.t
  // val y = DenseVector.horzcat(DenseVector.zeros[Double](x.size) + 1d, DenseVector.zeros[Double](x.size) + 2d)

  val covFuncG: Array[CovFunc] = Array(CovSEiso())
  val cofFuncGParams = Array(DenseVector(log(1), log(1)))

  val covFuncH: Array[CovFunc] = Array(CovSEiso(), CovSEiso())
  val covFuncHParams = Array(DenseVector(log(1), log(1)), DenseVector(log(1), log(1)))

  @Test def test = {

    val modelParams = cogp(x, y, covFuncG, cofFuncGParams, covFuncH, covFuncHParams, l = 0.1, iterNum = 100)

    println("w: " + modelParams.w)
    println("beta: " + modelParams.beta)
    println(modelParams.g(0).u.m)
    println(modelParams.h(0).u.m)
    println(modelParams.h(1).u.m)
  }
}