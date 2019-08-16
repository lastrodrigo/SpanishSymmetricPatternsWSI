package scoring;
import edu.ucla.clustercomparison.JaccardIndex;
import java.util.Map;


public class JniusJaccardIndex extends JaccardIndex {

     public double evaluate(Map<String, Double> goldSenseRatings, Map<String, Double> testSenseRatings) {
         return this.evaluateInstance(goldSenseRatings, testSenseRatings, 0);
     }

}
