package dk.gp.cogp.svi.w

import breeze.linalg.DenseMatrix
import breeze.linalg.InjectNumericOps
import breeze.linalg.sum
import breeze.linalg.trace
import breeze.numerics.pow
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.wAm
import dk.gp.cov.utils.covDiag

object calcLBGradW {

  def apply(lb: LowerBound, y: DenseMatrix[Double]): DenseMatrix[Double] = {

    val model = lb.model
    val x = lb.x
    
    val w = model.w
    val beta = model.beta
    val u = model.g.map(_.u)
    val v = model.h.map(_.u)

    val dw = DenseMatrix.zeros[Double](w.rows, w.cols)

    for (i <- 0 until dw.rows) {
      
      val kZZ2 = lb.kZZi(i)
      val kXZ2 = lb.kXZi(i)
      val kZX2 = kXZ2.t

      val kZZ2inv = lb.kZZiInv(i)
      val Ai = kXZ2 * kZZ2inv
      val lambdaI = Ai.t * Ai

      for (j <- 0 until dw.cols) {

        val kXZ = lb.kXZj(j)
        val kZZ = lb.kZZj(j)
        val kXXDiag = covDiag(x, model.g(j).covFunc, model.g(j).covFuncParams)

        val kZZinv = lb.kZZjInv(j)

        val Aj = kXZ * kZZinv
        val lambdaJ = Aj.t * Aj

        //trace term
        val traceTerm = beta(i) * w(i, j) * trace(u(j).v * lambdaJ)

        //kTilde term
        /**
         * trace(ABC) = trace(CAB) or trace(ABC) = sum(sum(ab.*c',2))
         * https://www.ics.uci.edu/~welling/teaching/KernelsICS273B/MatrixCookBook.pdf,
         * https://github.com/trungngv/cogp/blob/master/libs/util/diagProd.m
         */
        val kZX = kXZ.t

        val kTildeDiagSum = sum(kXXDiag) - trace(kZX * kXZ * kZZinv)
        val tildeTerm = beta(i) * w(i, j) * kTildeDiagSum

        //log normal term
        val logNormalTerm1 = beta(i) * ((y(::, i) - wAm(i,lb) - Ai * v(i).m + w(i, j) * (Aj * u(j).m)).t * (Aj * u(j).m)) //wAm (j'!= j) = wAm - wAjmj 
        val logNormalTerm2 = beta(i) * w(i, j) * sum(pow(Aj * u(j).m, 2))
        val logNormalTerm = logNormalTerm1 - logNormalTerm2

        dw(i, j) = logNormalTerm - tildeTerm - traceTerm
      }

    }

    dw
  }
}