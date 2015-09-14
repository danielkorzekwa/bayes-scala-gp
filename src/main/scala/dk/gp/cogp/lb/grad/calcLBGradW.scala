package dk.gp.cogp.lb.grad

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

    val dw = lb.model.w.mapPairs {
      case ((i, j), w) =>
        logNormalTerm(i, j, lb, y) - tildeTerm(i, j, lb) - traceTerm(i, j, lb)
    }

    dw
  }

  private def traceTerm(i: Int, j: Int, lb: LowerBound): Double = {

    val beta = lb.model.beta
    val w = lb.model.w
    val g = lb.model.g

    val Aj = lb.calcAj(j)
    val lambdaJ = Aj.t * Aj

    //trace term
    val traceTerm = beta(i) * w(i, j) * trace(g(j).u.v * lambdaJ)
    traceTerm
  }

  private def tildeTerm(i: Int, j: Int, lb: LowerBound): Double = {

    val model = lb.model
    val beta = lb.model.beta
    val w = lb.model.w

    val kXZ = lb.kXZj(j)

    val kZZinv = lb.kZZjInv(j)

    val kXXDiag = covDiag(lb.x, model.g(j).covFunc, model.g(j).covFuncParams)

    /**
     * trace(ABC) = trace(CAB) or trace(ABC) = sum(sum(ab.*c',2))
     * https://www.ics.uci.edu/~welling/teaching/KernelsICS273B/MatrixCookBook.pdf,
     * https://github.com/trungngv/cogp/blob/master/libs/util/diagProd.m
     */
    val kZX = kXZ.t

    val kTildeDiagSum = sum(kXXDiag) - trace(kZX * kXZ * kZZinv)
    val tildeTerm = beta(i) * w(i, j) * kTildeDiagSum

    tildeTerm
  }

  private def logNormalTerm(i: Int, j: Int, lb: LowerBound, y: DenseMatrix[Double]): Double = {

    val Ai = lb.calcAi(i)
    val Aj = lb.calcAj(j)
    val beta = lb.model.beta
    val w = lb.model.w
    val g = lb.model.g
    val h = lb.model.h

    //log normal term
    val logNormalTerm1 = beta(i) * ((y(::, i) - wAm(i, lb) - Ai * h(i).u.m + w(i, j) * (Aj * g(j).u.m)).t * (Aj * g(j).u.m)) //wAm (j'!= j) = wAm - wAjmj 
    val logNormalTerm2 = beta(i) * w(i, j) * sum(pow(Aj * g(j).u.m, 2))
    val logNormalTerm = logNormalTerm1 - logNormalTerm2

    logNormalTerm
  }
}