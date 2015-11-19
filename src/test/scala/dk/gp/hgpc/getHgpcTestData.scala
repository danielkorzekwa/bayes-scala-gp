package dk.gp.hgpc

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg._
import java.io.File

object getHgpcTestData {

  /**
   * Returns (x,y,u)
   */
  def apply(): (DenseMatrix[Double], DenseVector[Double], DenseMatrix[Double]) = {
    val x = csvread(new File("src/test/resources/gpml/classification_x.csv"))
    val y = csvread(new File("src/test/resources/gpml/classification_y.csv")).toDenseVector

    val x1Idx = (0 until x.rows-1).filter(idx => idx % 2 == 0)
    val x2Idx = (0 until x.rows-1).filter(idx => idx % 2 == 1)
    val x1 = DenseMatrix.horzcat(DenseMatrix.zeros[Double](x1Idx.size, 1) + 1.0, x(x1Idx, ::))
    val x2 = DenseMatrix.horzcat(DenseMatrix.zeros[Double](x2Idx.size, 1) + 2.0, x(x2Idx, ::))
    val x3 = DenseMatrix.horzcat(DenseMatrix.zeros[Double](1, 1) + 3.0, x(x.rows - 1 to x.rows - 1, ::))

    val allX = DenseMatrix.vertcat(x1, x2, x3)
    val allY = DenseVector.vertcat(y(x1Idx).toDenseVector, y(x2Idx).toDenseVector, y(19 to 19))
    val u = DenseMatrix.horzcat(DenseMatrix.zeros[Double](y.size, 1) - 1.0, x)

    (allX, allY, u)

  }
}