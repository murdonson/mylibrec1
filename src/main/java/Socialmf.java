import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.model.TextDataModel;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.rating.MAEEvaluator;
import net.librec.eval.rating.RMSEEvaluator;
import net.librec.math.algorithm.Maths;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.DenseMatrix;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.SocialRecommender;

import java.util.List;

public class Socialmf extends SocialRecommender
{
    public float socialValue;
    @Override
    public void setup() throws LibrecException
    {
        super.setup();
        userFactors.init(1.0);
        itemFactors.init(1.0);
        socialValue=conf.getFloat("soicalValue",0.5f);
    }



    @Override
    protected void trainModel() throws LibrecException
    {
        for (int iter = 1; iter <=numIterations; iter++)
        {
            loss=0.0d;
            DenseMatrix tempUserFactors=new DenseMatrix(numUsers,numFactors);
            DenseMatrix tempItemFactors=new DenseMatrix(numItems,numFactors);

            //rating
            for (int userIdx = 0; userIdx < numUsers; userIdx++)
            {
                List<Integer> ilist=trainMatrix.getColumns(userIdx);
                for(int i:ilist)
                {
                    double predictRating= Maths.logistic(DenseMatrix.rowMult(userFactors,userIdx,itemFactors,i));
                    double error=predictRating-Maths.normalize(trainMatrix.get(userIdx,i),minRate,maxRate);
                    loss+=Math.pow(error,2);
                    double deriValue=Maths.logisticGradientValue(DenseMatrix.rowMult(userFactors,userIdx,itemFactors,i))*error;

                    for (int factorIdx = 0; factorIdx <numFactors ; factorIdx++)
                    {
                        double userFactorValue=userFactors.get(userIdx,factorIdx);
                        double itemFactorValue=itemFactors.get(i,factorIdx);
                        tempUserFactors.add(userIdx,factorIdx,itemFactorValue*deriValue+regUser*userFactorValue);
                        tempItemFactors.add(i,factorIdx,userFactorValue*deriValue+regItem*itemFactorValue);
                        loss+=regUser*Math.pow(userFactorValue,2)+regItem*Math.pow(itemFactorValue,2);
                    }

                }
            }



            // friends
            for (int userIdx = 0; userIdx < numUsers; userIdx++)
            {

                List<Integer> vguanzhulist=socialMatrix.getColumns(userIdx);// u的关注 v
                if(vguanzhulist.size()<1)
                {
                    continue;
                }

                double []socialSumFactors=new double[numFactors];
                for(int v:vguanzhulist)
                {
                    for(int factorIdx=0;factorIdx<numFactors;factorIdx++)
                    {
                        socialSumFactors[factorIdx]+=userFactors.get(v,factorIdx);
                    }
                }

                for(int factorIdx=0;factorIdx<numFactors;factorIdx++)
                {
                    double usrFactorValue=userFactors.get(userIdx,factorIdx);
                    double socialValue=vguanzhulist.size()>0?socialSumFactors[factorIdx]/vguanzhulist.size():0;
                    tempUserFactors.add(userIdx,factorIdx,socialValue*(usrFactorValue-socialValue));
                    loss+=socialValue*Math.pow((usrFactorValue-socialValue),2);
                }

                List<Integer> vfensilist=socialMatrix.getRows(userIdx);//u的粉丝 v

                for(int v:vfensilist)
                {
                    List<Integer> wlist=socialMatrix.getColumns(v); // v的关注 w

                    double []socialSumFactors1=new double[numFactors];
                    for(int w:wlist)
                    {
                        for(int factorIdx=0;factorIdx<numFactors;factorIdx++)
                        {
                            socialSumFactors1[factorIdx]+=userFactors.get(w,factorIdx);
                        }
                    }


                    for(int factorIdx=0;factorIdx<numFactors;factorIdx++)
                    {
                        double userFactorValue=userFactors.get(v,factorIdx);
                        double socialValue=wlist.size()>0?socialSumFactors1[factorIdx]/wlist.size():0;
                        tempUserFactors.add(userIdx,factorIdx,-socialValue*(userFactorValue-socialValue));
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


    @Override
    protected double predict(int userIdx, int itemIdx) throws LibrecException
    {
        return super.predict(userIdx, itemIdx);
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
        conf.set("rec.iterator.maximum", "100");
        conf.set("rec.user.regularization", "0.001");
        conf.set("rec.item.regularization", "0.001");
        conf.set("rec.learnrate.bolddriver","false");
        conf.set("rec.learnrate.decay","1.0");
        conf.set("rec.recommender.earlystop","false");
        conf.set("rec.recommender.verbose","true");
        conf.set("rec.factor.number","5");
        conf.set("rec.user.social.ratio","1");
        conf.set("data.splitter.ratio", "rating");
        conf.set("soicalValue", "1.0");
        conf.set("rec.random.seed","1");
        Randoms.seed(1);
        TextDataModel dataModel = new TextDataModel(conf);
        dataModel.buildDataModel();
        RecommenderContext context = new RecommenderContext(conf, dataModel);
        Socialmf recommender=new Socialmf();
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
