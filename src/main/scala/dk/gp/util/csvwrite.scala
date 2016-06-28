package dk.gp.util

import java.io.File
import breeze.linalg.Matrix
import breeze.io.CSVWriter

object csvwrite {
  
  
  
  def apply[T](fileString: String, mat: Matrix[T],
            separator: Char = ',',
            quote: Char = '\u0000',
            escape: Char = '\\',
            skipLines: Int = 0,
            header: String = ""): Unit = {

    val matrixRows = IndexedSeq.tabulate(mat.rows, mat.cols)(mat(_, _).toString)

    val data = if (header.isEmpty) matrixRows
    else header.split(",").toIndexedSeq +: matrixRows

    CSVWriter.writeFile(new File(fileString), data, separator, quote, escape)
  }
}