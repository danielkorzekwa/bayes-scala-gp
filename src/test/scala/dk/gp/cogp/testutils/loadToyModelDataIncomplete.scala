package dk.gp.cogp.testutils

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg._
import dk.gp.cogp.model.Task

object loadToyModelDataIncomplete {

  def apply(): Array[Task] = {
    val y0Filter = (x: Double) => (x < -7 || x > -3)
    val y1Filter = (x: Double) => (x < 4 || x > 8)

    val data = loadToyModelData(n = Int.MaxValue, y0Filter, y1Filter)

    data
  }
}