package dk.gp.util

import com.typesafe.scalalogging.slf4j.LazyLogging
import breeze.linalg.DenseVector
import breeze.stats._
/**
 * Based on https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/evaluation/RankingMetrics.scala
 *  (there is a bug in a code there, score variable is not used)
 *
 * For reference R impl:
 *
 * apk <- function(k, actual, predicted)
 * {
 *  score <- 0.0
 *  cnt <- 0.0
 *  for (i in 1:min(k,length(predicted)))
 *  {
 *    if (predicted[i] %in% actual && !(predicted[i] %in% predicted[0:(i-1)]))
 *    {
 *      cnt <- cnt + 1
 *      score <- score + cnt/i
 *    }
 *  }
 *  score <- score / min(length(actual), k)
 *  score
 * }
 */
object meanAveragePrecision extends LazyLogging {

  def apply(predictionAndLabels: Seq[(Array[Double], Array[Double])], k: Int): Double = {

    require(k > 0, "ranking position k should be positive")
    val apk = predictionAndLabels.map { case (pred, lab) => averagePrecision(pred, lab, k) }.toArray
    mean(DenseVector(apk))
  }
}