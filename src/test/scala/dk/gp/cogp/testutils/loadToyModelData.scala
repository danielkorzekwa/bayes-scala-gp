package dk.gp.cogp.testutils

import breeze.linalg.DenseMatrix
import breeze.linalg._
import java.io._
import dk.gp.cogp.model.Task

object loadToyModelData {

  def apply(n: Int = Int.MaxValue, y0Filter: (Double) => Boolean = (x) => true, y1Filter: (Double) => Boolean = (x) => true): Array[Task] = {

    val allData = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))
    val data = n match {
      case Int.MaxValue => allData
      case _            => allData(0 until n, ::)
    }

    val y0Idx = data(::, 0).findAll { x => y0Filter(x) }
    val data0 = data(y0Idx, ::).toDenseMatrix
    val task0 = Task(data0(::, 0 to 0), data0(::, 1))

    val y1Idx = data(::, 0).findAll { x => y1Filter(x) }
    val data1 = data(y1Idx, ::).toDenseMatrix
    val task1 = Task(data1(::, 0 to 0), data1(::, 2))

    Array(task0, task1)
  }
}