import com.google.common.cache.LoadingCache;
import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.model.TextDataModel;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.rating.MAEEvaluator;
import net.librec.eval.rating.RMSEEvaluator;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.DenseVector;
import net.librec.math.structure.MatrixEntry;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.SocialRecommender;
import net.librec.similarity.PCCSimilarity;
import net.librec.similarity.RecommenderSimilarity;

import java.util.List;
import java.util.concurrent.ExecutionException;

public class TrustSvd extends SocialRecommender
{
    public DenseMatrix impItemFactors;
    public DenseMatrix trusteeFactors;
    public String cacheSpec;

    public DenseVector trusterWeights;
    public DenseVector trusteeWeights;
    public DenseVector impItemWeights;

    public LoadingCache<Integer,List<Integer>> userItemsCache,userTrusteeCache;

    private DenseVector userBiases, itemBiases;
    protected double regBias;


    @Override
    public void setup() throws LibrecException
    {
        super.setup();
        impItemFactors=new DenseMatrix(numItems,numFactors);
        trusteeFactors=new DenseMatrix(numUsers,numFactors);
        userBiases=new DenseVector(numUsers);
        itemBiases=new DenseVector(numItems);

        impItemFactors.init(initMean,initStd);
        trusteeFactors.init(initMean,initStd);
        userBiases.init(initMean,initStd);
        itemBiases.init(initMean,initStd);

        trusterWeights=new DenseVector(numUsers);
        trusteeWeights=new DenseVector(numUsers);
        impItemWeights=new DenseVector(numItems);

        for (int userIdx = 0; userIdx <numUsers ; userIdx++)
        {
            //粉丝个数
            int count=socialMatrix.columnSize(userIdx);
            //  v的粉丝权重
            trusteeWeights.set(userIdx,count>0?Math.sqrt(1/count):1.0);
            // 关注个数
            count=socialMatrix.rowSize(userIdx);
            // u的关注权重
            trusterWeights.set(userIdx,count>0?Math.sqrt(1/count):1.0);
        }
        for (int itemIdx = 0; itemIdx <numItems ; itemIdx++)
        {
            // 购买itemIdx的用户个数
            int count=trainMatrix.columnSize(itemIdx);
            impItemWeights.set(itemIdx,count>0?Math.sqrt(1/count):1.0);
        }
        cacheSpec=cacheSpec = conf.get("guava.cache.spec", "maximumSize=200,expireAfterAccess=2m");
        userItemsCache=trainMatrix.rowColumnsCache(cacheSpec);
        userTrusteeCache=socialMatrix.rowColumnsCache(cacheSpec);
        regBias=conf.getDouble("bias",1.2);

    }

    @Override
    protected void trainModel() throws LibrecException
    {
        for (int iter = 1; iter <=numIterations; iter++)
        {
            loss=0;
            DenseMatrix tempUserFactors=new DenseMatrix(numUsers,numFactors);
            DenseMatrix trusteeTempFactors=new DenseMatrix(numUsers,numFactors);
            for(MatrixEntry entry:trainMatrix)
            {
                int userIdx=entry.row();
                int itemIdx=entry.column();
                double rating=entry.get();
                double predictRating=globalMean+DenseMatrix.rowMult(userFactors,userIdx,itemFactors,itemIdx);
                List<Integer> impItemsList=null;
                try
                {
                    impItemsList=userItemsCache.get(userIdx);

                } catch (ExecutionException e)
                {
                    e.printStackTrace();
                }

                if(impItemsList.size()>0)
                {
                    double sum=0;
                    for(int imp:impItemsList)
                    {
                        sum+=DenseMatrix.rowMult(impItemFactors,imp,itemFactors,itemIdx);
                    }
                    predictRating+=sum/Math.sqrt(impItemsList.size());
                }

                List<Integer> trustedList=null;

                try
                {
                    trustedList=userTrusteeCache.get(userIdx);
                } catch (ExecutionException e)
                {
                    e.printStackTrace();
                }
                if(trustedList.size()>0)
                {
                    double sum=0;
                    for(int v:trustedList)
                    {
                        sum+=DenseMatrix.rowMult(trusteeFactors,v,itemFactors,itemIdx);
                    }
                    predictRating+=sum/Math.sqrt(trustedList.size());
                }
                double userBiasValue = userBiases.get(userIdx);
                double itemBiasValue = itemBiases.get(itemIdx);
                predictRating+=userBiasValue+itemBiasValue;
                double error=predictRating-rating;
                loss+=Math.pow(error,2);
                double userWeight=1/Math.sqrt(impItemsList.size());
                double trustedWeight=1/Math.sqrt(trustedList.size());
                double itemWeight=impItemWeights.get(itemIdx);
                double sgd=error+regBias*userWeight*userBiasValue;
                userBiases.add(userIdx,-learnRate*sgd);
                sgd=error+regBias*itemWeight*itemBiasValue;
                itemBiases.add(itemIdx,-learnRate*sgd);
                loss+=regBias*(userWeight*Math.pow(userBiasValue,2)+itemWeight*Math.pow(itemBiasValue,2));


                double []sumImpItemFactors=new double[numFactors];

                    for (int factorIdx = 0; factorIdx <numFactors ; factorIdx++)
                    {
                        for(int y:impItemsList)
                        {
                            sumImpItemFactors[factorIdx] += impItemFactors.get(y, factorIdx);
                        }
                        sumImpItemFactors[factorIdx]=impItemsList.size()>0?sumImpItemFactors[factorIdx]*userWeight:1.0;
                    }

                double []sumTrusteeFactors=new double[numFactors];
                for (int factorIdx = 0; factorIdx <numFactors ; factorIdx++)
                {
                    double sum=0;
                    for(int t:trustedList)
                    {
                        sum+=trusteeFactors.get(t,factorIdx);
                    }
                    sumTrusteeFactors[factorIdx]=trustedList.size()>0?sum/trustedList.size():1.0;
                }

                for (int factorIdx = 0; factorIdx <numFactors ; factorIdx++)
                {
                    double userFactorValue=userFactors.get(userIdx,factorIdx);
                    double itemFactorValue=itemFactors.get(itemIdx,factorIdx);
                    double deltaUser=error*itemFactorValue+regBias*userWeight;

                    tempUserFactors.add(userIdx,factorIdx,deltaUser);


                    double deltaItem=error*(userFactorValue+userWeight*sumImpItemFactors[factorIdx]+sumTrusteeFactors[factorIdx])+
                            regBias*impItemWeights.get(itemIdx);
                    itemFactors.add(itemIdx,factorIdx,-learnRate*deltaItem);


                    loss += regUser * userWeight * userFactorValue * userFactorValue
                            + regItem * itemWeight * itemFactorValue * itemFactorValue;
                    double ysgd=0;
                    for(int i:impItemsList)
                    {
                        double impItemFactorValue=impItemFactors.get(i,factorIdx);
                         ysgd+= error*userWeight*itemFactorValue+regBias*impItemWeights.get(i)*impItemFactorValue;
                        impItemFactors.add(i,factorIdx,-learnRate*ysgd);
                        loss+=regBias*impItemWeights.get(i)*Math.pow(impItemFactorValue,2);
                    }

                    double vsgd=0;
                    for(int v:trustedList)
                    {
                        double itemFactorvalue=itemFactors.get(itemIdx,factorIdx);
                        double wFactorvalue=trusteeFactors.get(v,factorIdx);
                        vsgd+=error*trusteeWeights.get(userIdx)*itemFactorValue+regBias*trusterWeights.get(v)*wFactorvalue;
                        trusteeTempFactors.add(v,factorIdx,vsgd);
                        loss+=regBias*trusterWeights.get(v)*wFactorvalue*wFactorvalue;
                    }

                }
            }


            for(MatrixEntry entry:socialMatrix)
            {
                int userIdx=entry.row();
                int socialUserIdx=entry.column();
                double value=entry.get();
                double error=DenseMatrix.rowMult(trusteeFactors,socialUserIdx,userFactors,userIdx)-value;
                double sgd=0;
                double sgd1=0;
                for (int factorIdx = 0; factorIdx <numFactors ; factorIdx++)
                {
                    double userFactorValue=userFactors.get(userIdx,factorIdx);
                    sgd+=regSocial*error*trusteeFactors.get(socialUserIdx,factorIdx)+regSocial*trusterWeights.get(userIdx)*userFactorValue;
                    tempUserFactors.add(userIdx,factorIdx,sgd);
                    sgd1+=regSocial*error*userFactorValue;
                    trusteeTempFactors.add(socialUserIdx,factorIdx,sgd1);
                }
            }


            userFactors.addEqual(tempUserFactors.scale(-learnRate));
            trusteeFactors.addEqual(trusteeTempFactors.scale(-learnRate));
            loss *= 0.5d;
            if (isConverged(iter) && earlyStop) {
                break;
            }
            updateLRate(iter);

        }
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
        conf.set("rec.user.regularization", "1.2");
        conf.set("rec.item.regularization", "1.2");
        conf.set("rec.learnrate.bolddriver","false");
        conf.set("rec.learnrate.decay","1.0");
        conf.set("rec.recommender.earlystop","false");
        conf.set("rec.recommender.verbose","true");
        conf.set("rec.factor.number","10");
        conf.set("data.splitter.ratio", "rating");
        conf.set("data.splitter.trainset.ratio", "0.8");
        conf.set("rec.social.regularization","0.9");
       conf.set("bias","1.2");

        Randoms.seed(1);
        TextDataModel dataModel = new TextDataModel(conf);
        try
        {
            dataModel.buildDataModel();
        } catch (LibrecException e)
        {
            e.printStackTrace();
        }
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



    @Override
    protected double predict(int userIdx, int itemIdx, boolean bounded) throws LibrecException
    {
        double predictRating = predict(userIdx, itemIdx);
        return predictRating;
    }


    @Override
    protected double predict(int userIdx, int itemIdx) throws LibrecException
    {
        double predictRating = globalMean + userBiases.get(userIdx) + itemBiases.get(itemIdx) + DenseMatrix.rowMult(userFactors, userIdx, itemFactors, itemIdx);

        //the implicit influence of items rated by user in the past on the ratings of unknown items in the future.
        List<Integer> userItemsList = null;
        try {
            userItemsList = userItemsCache.get(userIdx);
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
        if (userItemsList.size() > 0) {
            double sum = 0;
            for (int userItemIdx : userItemsList)
                sum += DenseMatrix.rowMult(impItemFactors, userItemIdx, itemFactors, itemIdx);
            predictRating += sum / Math.sqrt(userItemsList.size());
        }

        // the user-specific influence of users (trustees)trusted by user u
        List<Integer> trusteeList = null;
        try {
            trusteeList = userTrusteeCache.get(userIdx);
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
        if (trusteeList.size() > 0) {
            double sum = 0.0;
            for (int trusteeIdx : trusteeList)
                sum += DenseMatrix.rowMult(trusteeFactors, trusteeIdx, itemFactors, itemIdx);

            predictRating += sum / Math.sqrt(trusteeList.size());
        }

        return predictRating;
    }
}
