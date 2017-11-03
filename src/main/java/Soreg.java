import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.model.TextDataModel;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.rating.MAEEvaluator;
import net.librec.eval.rating.RMSEEvaluator;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.MatrixEntry;
import net.librec.math.structure.SymmMatrix;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.SocialRecommender;
import net.librec.similarity.PCCSimilarity;
import net.librec.similarity.RecommenderSimilarity;

import java.util.List;

public class Soreg extends SocialRecommender
{
    public SymmMatrix simMatrix;

    public float regSocial;
    @Override
    public void setup() throws LibrecException
    {
        super.setup();
        userFactors.init(1.0);
        itemFactors.init(1.0);
        simMatrix=context.getSimilarity().getSimilarityMatrix();
        regSocial=conf.getFloat("social",0.8f);

        for(int userIdx=0;userIdx<numUsers;userIdx++)
        {
            for(int userSocialIdx=userIdx+1;userSocialIdx<numUsers;userSocialIdx++)
            {
                if(simMatrix.contains(userIdx,userSocialIdx))
                {
                    double sim=simMatrix.get(userIdx,userSocialIdx);
                    double mappingsim=(1.0+sim)/2;
                    simMatrix.set(userIdx,userSocialIdx,mappingsim);
                }
            }
        }

    }

    @Override
    protected void trainModel() throws LibrecException
    {

        for (int iter = 1; iter <=numIterations; iter++)
        {
            loss=0.0d;
            DenseMatrix tempUserFactors=new DenseMatrix(numUsers,numFactors);
            DenseMatrix tempItemFactors=new DenseMatrix(numItems,numFactors);


            // ratings
            for(MatrixEntry entry:trainMatrix)
            {
                int userIdx=entry.row();
                int itemIdx=entry.column();
                double rating=entry.get();

                for (int factorIdx = 0; factorIdx < numFactors; factorIdx++)
                {
                    double userFactorValue=userFactors.get(userIdx,factorIdx);
                    double itemFactorValue=itemFactors.get(itemIdx,factorIdx);

                    double error=predict(userIdx,itemIdx)-rating;
                    loss+=Math.pow(error,2);

                    tempUserFactors.add(userIdx,factorIdx,error*itemFactorValue+regUser*userFactorValue);
                    tempItemFactors.add(itemIdx,factorIdx,error*userFactorValue+regItem*itemFactorValue);

                    loss+=(Math.pow(userFactorValue,2)+Math.pow(itemFactorValue,2))*regUser;


                }
            }




            //friends

            for(int userIdx=0;userIdx<numUsers;userIdx++)
            {
                List<Integer> flist=socialMatrix.getColumns(userIdx);

                for(int f:flist)
                {
                    double sim=simMatrix.get(userIdx,f);
                    if(!Double.isNaN(sim))
                    {
                        for (int factorIdx = 0; factorIdx <numFactors ; factorIdx++)
                        {
                            double error=userFactors.get(userIdx,factorIdx)-userFactors.get(f,factorIdx);
                            tempUserFactors.add(userIdx,factorIdx,regSocial*sim*(error));
                            loss+=regSocial*sim*(Math.pow(error,2));
                        }
                    }

                }

                List<Integer> glist=socialMatrix.getRows(userIdx);
                for(int g:glist)
                {
                    double sim=simMatrix.get(userIdx,g);
                    if(!Double.isNaN(sim))
                    {
                        for (int factorIdx = 0; factorIdx < numFactors; factorIdx++)
                        {
                            double error=userFactors.get(userIdx,factorIdx)-userFactors.get(g,factorIdx);
                            tempUserFactors.add(userIdx,factorIdx,regSocial*sim*(error));
                            //loss+=regSocial*sim*Math.pow(error,2);
                        }
                    }
                }
            }



            userFactors = userFactors.add(tempUserFactors.scale(-learnRate));
            itemFactors = itemFactors.add(tempItemFactors.scale(-learnRate));
            loss *= 0.5d;
            if (isConverged(iter) && earlyStop) {
                break;
            }
            updateLRate(iter);
        }

    }

    protected double predict(int userIdx, int itemIdx, boolean bound) throws LibrecException {
        double predictRating = predict(userIdx, itemIdx);

        if (bound) {
            if (predictRating > maxRate) {
                predictRating = maxRate;
            } else if (predictRating < minRate) {
                predictRating = minRate;
            }
        }

        return predictRating;
    }





    public static void main(String[] args) throws LibrecException
    {
        Configuration conf=new Configuration();
        conf.set("dfs.data.dir","D:/librec-2.0.0/data");
        conf.set("data.input.path","filmtrust/rating");
        conf.set("data.appender.class","social");
        conf.set("data.appender.path","filmtrust/trust");
        conf.set("rec.iterator.learnrate", "0.001");
        conf.set("rec.iterator.learnrate.maximum", "0.01");
        conf.set("rec.iterator.maximum", "100");
        conf.set("rec.user.regularization", "0.001");
        conf.set("rec.item.regularization", "0.001");
        conf.set("rec.learnrate.bolddriver","false");
        conf.set("rec.learnrate.decay","1.0");
        conf.set("rec.recommender.earlystop","false");
        conf.set("rec.recommender.verbose","true");
        conf.set("rec.factor.number","10");
        conf.set("data.splitter.ratio", "rating");
        conf.set("data.splitter.trainset.ratio", "0.8");
        conf.set("social","0.1");
        conf.set("rec.recommender.similarities","user");
        conf.set("rec.similarity.class","pcc");
        conf.set("rec.similarity.shrinkage","10");

        Randoms.seed(1);
        TextDataModel dataModel = new TextDataModel(conf);
        dataModel.buildDataModel();
        RecommenderContext context = new RecommenderContext(conf, dataModel);
        RecommenderSimilarity similarity = new PCCSimilarity();
        similarity.buildSimilarityMatrix(dataModel);
        context.setSimilarity(similarity);
        Soreg recommender=new Soreg();
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
