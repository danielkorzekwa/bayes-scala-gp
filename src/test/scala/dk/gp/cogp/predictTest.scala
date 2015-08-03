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

class predictTest {

  val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
  val x = data(::, 0).toDenseMatrix.t
  val y = data(::, 1 to 2)

  //val x: DenseMatrix[Double] = DenseVector.rangeD(-10, 10, 1).toDenseMatrix.t
  //val y = DenseVector.horzcat(DenseVector.zeros[Double](x.size) + 1d, DenseVector.zeros[Double](x.size) + 2d)

  val covFuncG:Array[CovFunc] = Array(CovSEiso())
  val cofFuncGParams = Array(DenseVector(log(1), log(1)))

  val covFuncH:Array[CovFunc] = Array(CovSEiso(), CovSEiso())
  val covFuncHParams = Array(DenseVector(log(1), log(1)), DenseVector(log(1), log(1)))

  @Test def test = {

    val model = cogp(x, y, covFuncG, cofFuncGParams,covFuncH,covFuncHParams)

    val loglik = calcLBLoglik(model, x,y)
    println("LB loglik=" + loglik)

    val s = x
    val predictedY = predict(s,x, model)

    println(predictedY)
  }
}