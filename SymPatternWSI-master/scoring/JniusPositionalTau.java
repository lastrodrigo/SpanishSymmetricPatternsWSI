package scoring;
import edu.ucla.clustercomparison.PositionalKendallsTau;
import java.util.Map;


public class JniusPositionalTau extends PositionalKendallsTau {

     public double evaluate(Map<String, Double> goldSenseRatings, Map<String, Double> testSenseRatings) {
         return this.evaluateInstance(goldSenseRatings, testSenseRatings, 0);
     }

}
