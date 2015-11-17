package dk.gp.gpc

import breeze.linalg.DenseMatrix
import dk.gp.cov.CovFunc
import breeze.linalg.DenseVector

case class GpcModel(x:DenseMatrix[Double],y:DenseVector[Double],covFunc:CovFunc,covFuncParams:DenseVector[Double],mean:Double=0)