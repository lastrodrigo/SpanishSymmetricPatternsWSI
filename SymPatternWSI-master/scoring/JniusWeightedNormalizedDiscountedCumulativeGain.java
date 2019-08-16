package scoring;
import edu.ucla.clustercomparison.WeightedNormalizedDiscountedCumulativeGain;
import java.util.Map;


public class JniusWeightedNormalizedDiscountedCumulativeGain extends WeightedNormalizedDiscountedCumulativeGain {

     public double evaluate(Map<String, Double> goldSenseRatings, Map<String, Double> testSenseRatings) {
         return this.evaluateInstance(goldSenseRatings, testSenseRatings, 0);
     }

}
