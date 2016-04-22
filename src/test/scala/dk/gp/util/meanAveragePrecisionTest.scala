package dk.gp.util

import org.junit._
import Assert._

class meanAveragePrecisionTest {

  @Test def test = {

    val predictionsAndLabels = Array(
      (Array(3, 4d), Array(4d)),
      (Array(6, 7, 8, 5d), Array(5d)))

    assertEquals(0, meanAveragePrecision(predictionsAndLabels, k = 1), 0.0001)
    assertEquals(0.25, meanAveragePrecision(predictionsAndLabels, k = 2), 0.0001)
    assertEquals(0.25, meanAveragePrecision(predictionsAndLabels, k = 3), 0.0001)
    assertEquals(0.375, meanAveragePrecision(predictionsAndLabels, k = 4), 0.0001)
    assertEquals(0.375, meanAveragePrecision(predictionsAndLabels, k = 5), 0.0001)
  }
}