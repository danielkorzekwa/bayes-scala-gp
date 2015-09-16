package dk.gp.cogp.testutils

import breeze.linalg.DenseMatrix
import breeze.linalg._
import java.io._

object loadToyModelData {
  
  def apply():DenseMatrix[Double] = {
    csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv")) //(0 to 40, ::)
  }
}