package dk.gp.cogp.svi.w

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.trace
import dk.gp.math.MultivariateGaussian
import breeze.linalg.inv
import breeze.linalg.sum
import breeze.numerics.pow
import dk.gp.math.MultivariateGaussian
import breeze.linalg._

object calcLBGradW {
 
  
  
  def apply(w: DenseMatrix[Double], beta: DenseVector[Double], u: Array[MultivariateGaussian], v: Array[MultivariateGaussian], kXZ: DenseMatrix[Double], kZZ: DenseMatrix[Double], kXXDiag: DenseVector[Double],
            y: DenseMatrix[Double]): DenseMatrix[Double] = {

    val dw = DenseMatrix.zeros[Double](w.rows, w.cols)

    for (i <- 0 until dw.rows) {
      
      val Aj = kXZ * inv(kZZ)
      val wAm =  (0 until dw.cols).foldLeft(DenseVector.zeros[Double](kXZ.rows))((wAm, j) => wAm + w(i, j) * Aj * u(j).m)

      for (j <- 0 until dw.cols) {

        //trace term
        val Ai = Aj
        val lambdaJ = Aj.t * Aj
        
        val traceTerm = beta(i) * w(i, j) * trace(u(j).v * lambdaJ)

        //kTilde term
        /**
         * trace(ABC) = trace(CAB) or trace(ABC) = sum(sum(ab.*c',2))
         * https://www.ics.uci.edu/~welling/teaching/KernelsICS273B/MatrixCookBook.pdf,
         * https://github.com/trungngv/cogp/blob/master/libs/util/diagProd.m
         */
        val kZX = kXZ.t
                  
        val kTildeDiagSum = sum(kXXDiag) - trace(kZX * kXZ * inv(kZZ))
        val tildeTerm = beta(i) * w(i, j) * kTildeDiagSum

        //log normal term
        val logNormalTerm1 =  beta(i) * ((y(::, i) - wAm - Ai * v(i).m + w(i, j) * (Aj * u(j).m)).t * (Aj * u(j).m)) //wAm (j'!= j) = wAm - wAjmj 
        val logNormalTerm2 = beta(i) * w(i, j) * sum(pow(Aj * u(j).m, 2))
        val logNormalTerm = logNormalTerm1 - logNormalTerm2

        dw(i, j) = logNormalTerm - tildeTerm - traceTerm
      }

    }

    dw
  }
}