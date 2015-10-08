package dk.gp.cogp

import breeze.linalg.DenseMatrix
import dk.gp.cogp.model.CogpModel
import dk.gp.cogp.svi.stochasticUpdateCogpModel
import breeze.linalg.DenseVector
import dk.gp.cogp.lb.LowerBound
import com.typesafe.scalalogging.slf4j.LazyLogging
import dk.gp.cogp.lb.calcLBLoglik
import dk.gp.cogp.model.Task

object cogpTrain extends LazyLogging{
  
  def apply(tasks:Array[Task], model: CogpModel, iterNum: Int): CogpModel = {
    
    val lowerBound = LowerBound(model,tasks)
    
    val finalLB = (0 until iterNum).foldLeft(lowerBound) { 
      case (currLB, iter) => 
        val newLB = stochasticUpdateCogpModel(currLB,tasks)
          logger.info("iter=%d/%d, llh=%.3f".format(iter,iterNum,calcLBLoglik(newLB)))
          newLB
      }
    finalLB.model
  }
}