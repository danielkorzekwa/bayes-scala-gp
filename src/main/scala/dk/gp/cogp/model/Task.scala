package dk.gp.cogp.model

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

case class Task(x:DenseMatrix[Double],y:DenseVector[Double]) 

object Task {
  
  def apply(x:DenseVector[Double],y:DenseVector[Double]) = new Task(x.toDenseMatrix.t,y)
}