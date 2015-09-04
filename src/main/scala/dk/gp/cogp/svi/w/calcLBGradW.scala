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
import dk.gp.cogp.CogpModel
import dk.gp.cov.utils.covDiag
import dk.gp.math.invchol
import dk.gp.cogp.lb.LowerBound

object calcLBGradW {

  def apply(lowerBound: LowerBound, model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = {

    val w = model.w
    val beta = model.beta
    val u = model.g.map(_.u)
    val v = model.h.map(_.u)

    val dw = DenseMatrix.zeros[Double](w.rows, w.cols)

    for (i <- 0 until dw.rows) {

      val wAm = (0 until dw.cols).foldLeft(DenseVector.zeros[Double](x.rows)) { (wAm, j) =>

        val kXZ = lowerBound.kXZj(j)
        val kZZ = lowerBound.kZZj(j)
        val kZZinv = lowerBound.kZZjInv(j)

        val Aj = kXZ * kZZinv

        wAm + w(i, j) * Aj * u(j).m
      }

      val kZZ2 = lowerBound.kZZi(i)
      val kXZ2 = lowerBound.kXZi(i)
      val kZX2 = kXZ2.t

      val kZZ2inv = lowerBound.kZZiInv(i)
      val Ai = kXZ2 * kZZ2inv
      val lambdaI = Ai.t * Ai

      for (j <- 0 until dw.cols) {

        val z = model.g(j).z
        val kXZ = lowerBound.kXZj(j)
        val kZZ = lowerBound.kZZj(j)
        val kXXDiag = covDiag(x, model.g(j).covFunc, model.g(j).covFuncParams)

        val kZZinv = lowerBound.kZZjInv(j)

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
        val logNormalTerm1 = beta(i) * ((y(::, i) - wAm - Ai * v(i).m + w(i, j) * (Aj * u(j).m)).t * (Aj * u(j).m)) //wAm (j'!= j) = wAm - wAjmj 
        val logNormalTerm2 = beta(i) * w(i, j) * sum(pow(Aj * u(j).m, 2))
        val logNormalTerm = logNormalTerm1 - logNormalTerm2

        dw(i, j) = logNormalTerm - tildeTerm - traceTerm
      }

    }

    dw
  }
}