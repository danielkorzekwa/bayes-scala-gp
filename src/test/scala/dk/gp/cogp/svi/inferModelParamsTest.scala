package dk.gp.cogp.svi

import org.junit._
import org.junit.Assert._
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovSEiso
import scala.math._
import dk.gp.cogp.svi.inferModelParams
import dk.gp.cov.CovSEiso

class inferModelParamsTest {

  val x: DenseMatrix[Double] = DenseVector.rangeD(-10, 10, 1).toDenseMatrix.t
  val y = DenseVector.horzcat(DenseVector.zeros[Double](x.size) + 1d, DenseVector.zeros[Double](x.size) + 2d)

  val covFunc = CovSEiso(log(1), log(1))

  @Test def test {
    val modelParams = inferModelParams(x, y, covFunc, l = 0.1, iterNum = 10)
    println(modelParams.u(0).m)
    println(modelParams.v(0).m)
    println(modelParams.v(1).m)
  }
}