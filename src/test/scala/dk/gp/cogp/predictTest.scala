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
import dk.gp.cogp.svi.cogp
import dk.gp.cogp.svi.stochasticUpdateLB

class predictTest {

  val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
  val x = data(::, 0).toDenseMatrix.t
  val y = data(::, 1 to 2)

  //val x: DenseMatrix[Double] = DenseVector.rangeD(-10, 10, 1).toDenseMatrix.t
  //val y = DenseVector.horzcat(DenseVector.zeros[Double](x.size) + 1d, DenseVector.zeros[Double](x.size) + 2d)

  val covFuncG: Array[CovFunc] = Array(CovSEiso())
  val covFuncGParams = Array(DenseVector(log(1), log(1)))

  val covFuncH: Array[CovFunc] = Array(CovSEiso(), CovSEiso())
  val covFuncHParams = Array(DenseVector(log(1), log(1e-10)), DenseVector(log(1), log(1e-10)))

  @Test def test = {

    val initialModel = CogpModel(x, y, covFuncG, covFuncGParams, covFuncH, covFuncHParams)

    val finalModel = (1 to 20).foldLeft(initialModel) {
      case (currentModel, i) =>
        val newModel = stochasticUpdateLB(currentModel, x, y)

        val loglik = calcLBLoglik(newModel, x, y)
        println("LB loglik=" + loglik)

        newModel
    }

    val s = x
    val predictedY = predict(s, x, finalModel)

    println(predictedY)
  }
}