package dk.gp.cogp.lb

import breeze.linalg.DenseVector

object wAm {

  def apply(i: Int, lb: LowerBound): DenseVector[Double] = {
    val wAm = (0 until lb.model.g.size).foldLeft(DenseVector.zeros[Double](lb.x.rows)) { (wAm, j) =>

      val Aj = lb.calcAj(j)
      wAm + lb.model.w(i, j) * Aj * lb.model.g(j).u.m
    }

    wAm
  }
}