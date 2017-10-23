import com.google.common.cache.LoadingCache;
import com.google.common.collect.Table;
import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.model.TextDataModel;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.ranking.AUCEvaluator;
import net.librec.eval.ranking.NormalizedDCGEvaluator;
import net.librec.eval.ranking.PrecisionEvaluator;
import net.librec.eval.ranking.RecallEvaluator;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.DenseVector;
import net.librec.math.structure.MatrixEntry;
import net.librec.recommender.MatrixFactorizationRecommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.cf.ranking.BPRRecommender;

import java.util.*;
import java.util.concurrent.ExecutionException;


public class FismAuc extends MatrixFactorizationRecommender {

    DenseMatrix itemfactors2=null;
    DenseVector itemBiases=null;

    protected static String cacheSpec;
    public double rho;

    public double alpha;
    public double lRate;
    public double gamma;
    public double beta;
    protected LoadingCache<Integer, List<Integer>> userItemsCache;
    @Override
    protected void setup() throws LibrecException {

        super.setup();
        itemFactors.init(0,0.01);
        itemBiases=new DenseVector(numItems);
        itemfactors2=new DenseMatrix(numItems,numFactors);
        itemfactors2.init(0,0.01);
        itemBiases.init(0,0.01);
        cacheSpec = conf.get("guava.cache.spec", "maximumSize=200,expireAfterAccess=2m");
        userItemsCache = trainMatrix.rowColumnsCache(cacheSpec);

        // 硬编码  自己的变量
        rho=0.5;
        lRate=0.00001;
        gamma=0.1;
        beta=0.6;
        alpha=0.5;

    }

    protected List unratedSampledItems(List ratedlist,int size,int userId) throws Exception
    {
        int samplesize= (int) (size*rho);
        List<Integer> indices = Randoms.randInts(samplesize, 0, numItems);
        // 模仿 librec的写法 不是精准的抽样
        indices.removeAll(ratedlist);
        return indices;

    }


    @Override
    protected double predict(int userIdx, int itemIdx) throws LibrecException {
        double pred=itemBiases.get(itemIdx);
        List<Integer> ratedItems=null;
        double sum=0;
        int count=0;
        try {
            ratedItems = userItemsCache.get(userIdx);
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
        for(int j:ratedItems)
        {
            if(j!=itemIdx)
            {
               sum+= DenseMatrix.rowMult(itemfactors2, itemIdx, itemFactors, j);
               count++;
            }
        }
        double wu = count > 0 ? Math.pow(count, -alpha) : 0;
        return pred+wu;
    }

    @Override
    protected void trainModel() throws LibrecException {
        long start=System.currentTimeMillis();
        for(int iter=0;iter<=numIterations;iter++)
        {
            loss = 0.0d;
            for(int userId=0;userId<numUsers;userId++)
            {
                List<Integer> ratedItems= null;
                List<Integer> unratedItems=null;
                int rateItemsSize=0;
                try {
                    ratedItems = userItemsCache.get(userId);
                   rateItemsSize=ratedItems.size();
                    if(rateItemsSize<2)
                    {
                        rateItemsSize=2;
                    }
                    unratedItems=unratedSampledItems(ratedItems,rateItemsSize,userId);
                } catch (Exception e) {
                    e.printStackTrace();
                }

                for(int i:ratedItems)
                {
                    DenseVector t=new DenseVector(numFactors);
                    t.init(0);
                    DenseVector x=new DenseVector(numFactors);
                    x.init(0);
                    for(int j:ratedItems)
                    {
                        if (j!=i)
                        {
                            t.add(itemfactors2.row(j));
                        }

                    }
                    t.scale(Math.pow(rateItemsSize-1,-alpha));
                    double rui=itemBiases.get(i)+t.inner(itemFactors.row(i));
                    double bi = itemBiases.get(i);
                    // 注意 两个j不一样哦
                    for(int j:unratedItems)
                    {
                        double ruj=itemBiases.get(j)+t.inner(itemFactors.row(j));
                        double e= trainMatrix.get(userId,i)-trainMatrix.get(userId,j)-rui+ruj;
                        loss+=e*e;

                        double bj = itemBiases.get(j);
                        loss+=gamma*(bi*bi+bj*bj);

                        // update bi  bj
                        itemBiases.add(i, lRate * (e - gamma * bi));
                        itemBiases.add(j, lRate * (e - gamma * bj));
                        for(int factorIdx=0;factorIdx<numFactors;factorIdx++)
                        {
                            double itemfactorvaluei=itemFactors.get(i,factorIdx);
                            double itemfactorvaluej=itemFactors.get(j,factorIdx);

                            // update qi qj    准确说是 qif qjf
                            itemFactors.add(i,factorIdx,lRate*(t.get(factorIdx)*e-beta*itemfactorvaluei));
                            itemFactors.add(j,factorIdx,-lRate*(t.get(factorIdx)*e+beta*itemfactorvaluej));

                            loss+=beta*(itemfactorvaluej*itemfactorvaluej+itemfactorvaluei*itemfactorvaluei);

                        }
                        x.add((itemFactors.row(i).minus(itemFactors.row(j))).scale(e));
                    }
                    for(int j:ratedItems)
                    {
                        if(j!=i)
                        {
                            for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                                double valuej = itemfactors2.get(j, factorIdx);
                                //update pj
                                itemfactors2.add(j, factorIdx, lRate * (Math.pow(rho, -1) * x.get(factorIdx)*Math.pow(rateItemsSize - 1, -alpha) - beta * valuej));
                                loss += beta * (valuej * valuej);
                            }
                        }
                    }

                }


            }
            loss*=0.5;
            System.out.println("iter: "+iter+" loss: "+loss);
            if (isConverged(iter) && earlyStop) {
                break;
            }
            //updateLRate(iter);

        }


        long end=System.currentTimeMillis();
        System.out.println("耗时: "+(end-start)/1000+"秒");

    }


    public static void main(String[] args) throws LibrecException {
       Configuration conf=new Configuration() ;
        conf.set("dfs.data.dir","D:/librec-2.0.0/data");
        conf.set("data.input.path","filmtrust/rating");
        conf.set("rec.iterator.learnrate", "0.01");
        //conf.set("rec.recommender.similarity.key" ,"item");
        conf.set("rec.iterator.learnrate.maximum", "0.01");
        conf.set("rec.iterator.maximum", "20");
        conf.set("rec.user.regularization", "0.001");
        conf.set("rec.item.regularization", "0.001");
        conf.set("rec.factor.number", "10");
        conf.set("rec.recommender.isranking", "true");
        conf.set("data.splitter.ratio", "rating");
        conf.set("data.splitter.trainset.ratio", "0.8");
        TextDataModel dataModel = new TextDataModel(conf);
        dataModel.buildDataModel();
        RecommenderContext context = new RecommenderContext(conf, dataModel);
        FismAuc recommender=new FismAuc();
        recommender.setContext(context);
        RecommenderEvaluator evaluator = new AUCEvaluator();
        RecommenderEvaluator evaluator1=new NormalizedDCGEvaluator();
        RecommenderEvaluator evaluator2=new PrecisionEvaluator();
        RecommenderEvaluator evaluator3=new RecallEvaluator();

        evaluator.setTopN(5);
        evaluator1.setTopN(5);
        evaluator2.setTopN(5);
        evaluator3.setTopN(5);

        double auc=0,ndcg=0,prec=0,rec=0;
        for(int i=0;i<1;i++)
        {
            recommender.recommend(context);
            auc+=recommender.evaluate(evaluator);
            ndcg+=recommender.evaluate(evaluator1);
            prec+=recommender.evaluate(evaluator2);
            rec+=recommender.evaluate(evaluator3);

        }
        System.out.println("AUC:" + auc/1);
        System.out.println("NDCG:" + ndcg/1);
        System.out.println("PRECISION:" + prec/1);
        System.out.println("REALL:" + rec/1);
    }




}
