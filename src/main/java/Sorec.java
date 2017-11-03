import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.model.TextDataModel;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.ranking.AUCEvaluator;
import net.librec.eval.ranking.NormalizedDCGEvaluator;
import net.librec.eval.ranking.PrecisionEvaluator;
import net.librec.eval.ranking.RecallEvaluator;
import net.librec.eval.rating.MAEEvaluator;
import net.librec.eval.rating.RMSEEvaluator;
import net.librec.math.algorithm.Maths;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.MatrixEntry;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.SocialRecommender;
import java.util.ArrayList;
import java.util.List;

public class Sorec extends SocialRecommender {
    public DenseMatrix userSoicalFactors;
    public List<Integer> indegrees;
    public List<Integer> outdegrees;
    private float regRateSocial, regUserSocial;
    @Override
    public void setup() throws LibrecException {
        super.setup();
        userFactors.init(1.0);
        itemFactors.init(1.0);
        userSoicalFactors=new DenseMatrix(numUsers,numFactors);
        userSoicalFactors.init(1.0);
        indegrees=new ArrayList<Integer>();
        outdegrees=new ArrayList<Integer>();

       for(int userIdx=0;userIdx<numUsers;userIdx++)
       {
           int in=trainMatrix.columnSize(userIdx);
           int out=trainMatrix.rowSize(userIdx);
            indegrees.add(in);
            outdegrees.add(out);
       }

        regRateSocial = conf.getFloat("rec.rate.social.regularization", 0.01f);
        regUserSocial = conf.getFloat("rec.user.social.regularization", 0.01f);

    }

    protected void trainModel() throws LibrecException
    {
        for(int iter=1;iter<numIterations;iter++)
            {
                loss=0.0d;
                DenseMatrix tempUserFactors=new DenseMatrix(numUsers,numFactors);
                DenseMatrix tempItemFactors=new DenseMatrix(numItems,numFactors);
                DenseMatrix tempSocialUserFactors=new DenseMatrix(numUsers,numFactors);
                //rating
                for(MatrixEntry entry:trainMatrix)
               {
                   int userId=entry.row();
                   int itemId=entry.column();
                   double rate=entry.get();

                   double predictrating=predict(userId,itemId);
                   double error=Maths.logistic(predictrating)-Maths.normalize(rate,minRate,maxRate);

                   loss+=error*error;

                   for(int factorIdx=0;factorIdx<numFactors;factorIdx++)
                   {
                       //计算 ui和vj
                       double userFactorvalue=userFactors.get(userId,factorIdx);
                       double itemFactorvalue=itemFactors.get(itemId,factorIdx);
                       // 梯度下降
                       tempUserFactors.add(userId,factorIdx,Maths.logisticGradientValue(predictrating)*error*itemFactorvalue+regUser*userFactorvalue);
                       tempItemFactors.add(itemId,factorIdx,Maths.logisticGradientValue(predictrating)*error*userFactorvalue+regItem*itemFactorvalue);
                       loss+=regUser*userFactorvalue*userFactorvalue+regItem*itemFactorvalue*itemFactorvalue;

                   }

               }
                //friends
                for(MatrixEntry entry:socialMatrix)
                {
                    int userId=entry.row();
                    int socialUserId=entry.column();
                    double socialvalue=entry.get();
                    if (socialvalue <= 0)
                    {
                        continue;
                    }


                    double socialpredictrating=DenseMatrix.rowMult(userFactors,userId,userSoicalFactors,socialUserId);
                    int vk=indegrees.get(socialUserId);
                    int vi=outdegrees.get(userId);

                    double realsoicalvalue=Math.sqrt(vk/(vk+vi+0.0))*socialvalue;
                    double error=Maths.logistic(socialpredictrating)-realsoicalvalue;
                    loss+=regRateSocial*error*error;
                    for(int factorIdx=0;factorIdx<numFactors;factorIdx++)
                    {
                        double userSocialFactorvalue=userSoicalFactors.get(socialUserId,factorIdx);
                        double userFactorvalue=userFactors.get(userId,factorIdx);
                        tempUserFactors.add(userId,factorIdx,regRateSocial*Maths.logisticGradientValue(socialpredictrating)*error*userSocialFactorvalue);
                        tempSocialUserFactors.add(socialUserId,factorIdx,regRateSocial*Maths.logisticGradientValue(socialpredictrating)*error*userFactorvalue+regUserSocial*userSocialFactorvalue);
                        loss+=regRateSocial*userSocialFactorvalue*userSocialFactorvalue;
                    }
                }

                userFactors=userFactors.add(tempUserFactors.scale(-learnRate));
                itemFactors=itemFactors.add(tempItemFactors.scale(-learnRate));
                userSoicalFactors=userSoicalFactors.add(tempSocialUserFactors.scale(-learnRate));
                loss*=0.5d;
                System.out.println("iter:"+iter+"loss:"+loss);
                if (isConverged(iter) && earlyStop) {
                    break;
                }
                //updateLRate(iter);

            }

    }


    public static void main(String[] args) throws LibrecException
    {
        Configuration conf=new Configuration();
        conf.set("dfs.data.dir","D:/librec-2.0.0/data");
        conf.set("data.input.path","filmtrust/rating");
        conf.set("data.appender.class","social");
        conf.set("data.appender.path","filmtrust/trust");
        conf.set("rec.iterator.learnrate", "0.01");
        conf.set("rec.iterator.learnrate.maximum", "-1");
        conf.set("rec.iterator.maximum", "200");
        conf.set("rec.user.regularization", "0.001");
        conf.set("rec.item.regularization", "0.001");
        conf.set("rec.learnrate.bolddriver","false");
        conf.set("rec.learnrate.decay","1.0");
        conf.set("rec.recommender.earlystop","false");
        conf.set("rec.recommender.verbose","true");
        conf.set("rec.factor.number","5");
        conf.set("rec.rate.social.regularization", "0.01");
        conf.set("rec.user.social.regularization", "0.01");
        conf.set("data.splitter.ratio", "rating");
        conf.set("data.splitter.trainset.ratio", "0.8");
        Randoms.seed(1);
        TextDataModel dataModel = new TextDataModel(conf);
        dataModel.buildDataModel();
        RecommenderContext context = new RecommenderContext(conf, dataModel);
        Sorec recommender=new Sorec();
        recommender.setContext(context);
        RecommenderEvaluator evaluator = new MAEEvaluator();
        RecommenderEvaluator evaluator1=new RMSEEvaluator();
        double mae=0,rmse=0;
        for(int i=0;i<1;i++)
        {
            recommender.recommend(context);
            mae+=recommender.evaluate(evaluator);
            rmse+=recommender.evaluate(evaluator1);
        }
        System.out.println("MAE:" + mae/1);
        System.out.println("RMSE:" + rmse/1);

    }
}
