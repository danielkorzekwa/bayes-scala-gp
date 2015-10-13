package dk.gp.cogp.lb.grad

import breeze.linalg.DenseMatrix
import breeze.linalg.InjectNumericOps
import breeze.linalg.sum
import breeze.linalg.trace
import breeze.numerics.pow
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.wAm
import dk.gp.cov.utils.covDiag
import dk.gp.math.diagProd

object calcLBGradW {

  def apply(lb: LowerBound): DenseMatrix[Double] = {

    val dw = lb.model.w.mapPairs {
      case ((i, j), w) =>
        logNormalTerm(i, j, lb) - tildeTerm(i, j, lb) - traceTerm(i, j, lb)
    }

    dw
  }

  private def traceTerm(i: Int, j: Int, lb: LowerBound): Double = {

    val beta = lb.model.beta
    val w = lb.model.w
    val g = lb.model.g

    val Aj = lb.Aj(i, j)
    val lambdaJ = lb.lambdaJ(i, j)

    //trace term
    val traceTerm = beta(i) * w(i, j) * sum(diagProd(g(j).u.v,lambdaJ))
    traceTerm
  }

  private def tildeTerm(i: Int, j: Int, lb: LowerBound): Double = {

    val beta = lb.model.beta
    val w = lb.model.w
    val tildeTerm = beta(i) * w(i, j) * lb.tildeQ(i, j)
    tildeTerm
  }

  private def logNormalTerm(i: Int, j: Int, lb: LowerBound): Double = {

    val Ai = lb.calcAi(i)
    val Aj = lb.Aj(i, j)
    val beta = lb.model.beta
    val w = lb.model.w
    val g = lb.model.g
    val h = lb.model.h
    val y = lb.yi(i)

    //log normal term

    val logNormalTerm1 = beta(i) * ((y - wAm(i, lb) - Ai * h(i).u.m + w(i, j) * (Aj * g(j).u.m)).t * (Aj * g(j).u.m)) //wAm (j'!= j) = wAm - wAjmj 
    val logNormalTerm2 = beta(i) * w(i, j) * sum(pow(Aj * g(j).u.m, 2))
    val logNormalTerm = logNormalTerm1 - logNormalTerm2

    logNormalTerm
  }
}