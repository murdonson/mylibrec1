import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.model.TextDataModel;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.rating.MAEEvaluator;
import net.librec.eval.rating.RMSEEvaluator;
import net.librec.math.algorithm.Maths;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.SparseVector;
import net.librec.math.structure.VectorEntry;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.SocialRecommender;
import net.librec.recommender.context.rating.RSTERecommender;

import java.util.List;

public class RSTE extends SocialRecommender
{
    public float userSocialRatio;
    @Override
    public void setup() throws LibrecException
    {
        super.setup();
        userFactors.init(1.0);
        itemFactors.init(1.0);
        userSocialRatio = conf.getFloat("rec.user.social.ratio", 0.8f);

    }
    @Override
    protected double predict(int userIdx, int itemIdx) throws LibrecException
    {
        List<Integer> klist=socialMatrix.getColumns(userIdx);
        double sum=0;
        for(int k:klist)
        {
            sum+=DenseMatrix.rowMult(userFactors,k,itemFactors,itemIdx);
        }
        sum=klist.size()>0?sum/klist.size():0;
        double finaPredictlRating=userSocialRatio*DenseMatrix.rowMult(userFactors,userIdx,itemFactors,itemIdx)+
                (1-userSocialRatio)*sum;

        return finaPredictlRating;
    }

    @Override
    protected void trainModel() throws LibrecException
    {
        // 这一部分是我自己写的RSTE     死都不知道问题出在哪 为啥效果那么差
        for (int iter = 1; iter <= numIterations; iter++)
        {
            loss=0.0;
            // 每一次迭代的梯度
            DenseMatrix tempUserFactors=new DenseMatrix(numUsers,numFactors);
            DenseMatrix tempItemFactors=new DenseMatrix(numItems,numFactors);
            //ratings
            for (int userIdx = 0; userIdx < numUsers; userIdx++)
            {
                List<Integer> klist=socialMatrix.getColumns(userIdx);//用户k  i的关注集合
                List<Integer> jlist=trainMatrix.getColumns(userIdx);//商品j i的购买集合
                double []socialVector=new double[numFactors];
                for(int k:klist)
                {
                    for(int factorIdx=0;factorIdx<numFactors;factorIdx++) {
                        socialVector[factorIdx]+=(1-userSocialRatio)*
                                userFactors.get(k,factorIdx);
                    }
                }
                double rateSum=0.0;
                for(int j:jlist)
                {
                    rateSum=userSocialRatio*DenseMatrix.rowMult(userFactors,userIdx,itemFactors,j);
                    double socialSum=0.0;
                    for(int k:klist)
                    {
                        socialSum+=(1-userSocialRatio)*DenseMatrix.rowMult(userFactors,k,itemFactors,j);
                    }
                    socialSum=klist.size()>0?socialSum/klist.size():0;
                    double finalpredictrating=rateSum+socialSum;
                    double error= Maths.logistic(finalpredictrating)-Maths.normalize(trainMatrix.get(userIdx,j),0.5,maxRate);
                    double deriveValue=Maths.logisticGradientValue(finalpredictrating)*error;
                    loss+=error*error;
                    for (int factorIdx = 0; factorIdx <numFactors ; factorIdx++)
                    {
                        double ui=userFactors.get(userIdx,factorIdx);
                        double vj=itemFactors.get(j,factorIdx);
                        double socialVectorValue=klist.size()>0?socialVector[factorIdx]/klist.size():0;
                        tempUserFactors.add(userIdx,factorIdx,userSocialRatio*deriveValue*vj+regUser*ui);
                        tempItemFactors.add(j,factorIdx,deriveValue*(userSocialRatio*userFactors.get(userIdx,factorIdx)+socialVectorValue)+regItem*vj);
                        loss += regUser * ui * ui + regItem * vj * vj;
                    }
                }
            }
            //social
            for(int userIdx=0;userIdx<numUsers;userIdx++)
            {
                List<Integer> plist = socialMatrix.getRows(userIdx);
                for(int p:plist)
                {
                    if (p >= numUsers)
                        continue;

                    List<Integer> klist = socialMatrix.getColumns(p);

                    // 注意此处j是p买的东西
                    List<Integer> jlist=trainMatrix.getColumns(p);

                    for(int j:jlist)
                    {
                        double sum = 0;
                        for (int k : klist)
                        {
                            sum += (1 - userSocialRatio) * DenseMatrix.rowMult(userFactors, k, itemFactors, j);
                        }
                        sum=klist.size()>0?sum/klist.size():0;

                        double finalPredictRating=userSocialRatio*DenseMatrix.rowMult(userFactors,p,itemFactors,j)+sum;
                        double error=Maths.logistic(finalPredictRating)-Maths.normalize(trainMatrix.get(p,j),0.5,maxRate);
                        double deriveValue=Maths.logisticGradientValue(finalPredictRating)*error;
                        for(int factorIdx=0;factorIdx<numFactors;factorIdx++)
                        {
                            tempUserFactors.add(userIdx,factorIdx,(1-userSocialRatio)*deriveValue*itemFactors.get(j,factorIdx));
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
        conf.set("data.splitter.trainset.ratio", "0.8");
        conf.set("rec.random.seed","1");
        Randoms.seed(1);
        TextDataModel dataModel = new TextDataModel(conf);
        dataModel.buildDataModel();
        RecommenderContext context = new RecommenderContext(conf, dataModel);
        RSTE recommender=new RSTE();
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
