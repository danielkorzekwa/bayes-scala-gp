package dk.gp.util

import org.junit._
import Assert._

class averagePrecisionTest {
  
  @Test def test = {
    
    val ap = averagePrecision(Array(2d,1,1,4,5),Array(1d),k=5)
    assertEquals(0.5,ap,0.00001)
  }
}