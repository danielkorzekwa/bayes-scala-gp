package dk.gp.math

import breeze.linalg.DenseMatrix
import breeze.linalg.eig
import breeze.linalg._

object inveig {

  def apply(x: DenseMatrix[Double]) = {
    if (x.size == 1) 1d / x
    else {
      val eigM = eig(x)
      val invM = eigM.eigenvectors * diag(1.0 :/ eigM.eigenvalues) * eigM.eigenvectors.t
      invM
    }
  }
}