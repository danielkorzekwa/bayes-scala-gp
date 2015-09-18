package dk.gp.cogp.testutils

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg._

object loadToyModelDataIncomplete {

  def apply(): DenseMatrix[Double] = {
    val allData = loadToyModelData()

    val data = allData(*, ::).map { r =>
      val y0 = if (r(0) > -7 && r(0) < -3) Double.NaN else r(1)
      val y1 = if (r(0) > 4 && r(0) < 8) Double.NaN else r(2)
      DenseVector(r(0), y0, y1)
    }

    data
  }
}