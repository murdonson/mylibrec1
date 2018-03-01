import com.google.common.cache.LoadingCache;
import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.model.TextDataModel;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.ranking.AUCEvaluator;
import net.librec.eval.ranking.NormalizedDCGEvaluator;
import net.librec.eval.ranking.PrecisionEvaluator;
import net.librec.eval.ranking.RecallEvaluator;
import net.librec.math.algorithm.Maths;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.DenseVector;
import net.librec.math.structure.SymmMatrix;
import net.librec.recommender.MatrixFactorizationRecommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.cf.ranking.BPRRecommender;
import net.librec.similarity.BinaryCosineSimilarity;
import net.librec.similarity.PCCSimilarity;
import net.librec.similarity.RecommenderSimilarity;

import java.util.List;
import java.util.concurrent.ExecutionException;

public class PFMSM extends MatrixFactorizationRecommender
{
    private DenseMatrix W;
    private double lamdaS;
    private double alpha;
    private String cacheSpec;
    private DenseVector itemBias;

    private SymmMatrix itemSimilarityMatrix;

    private LoadingCache<Integer, List<Integer>> userItemsCache, itemUsersCache;

    @Override
    protected void setup() throws LibrecException
    {
        super.setup();
        lamdaS=conf.getDouble("lamdaS",0.5);
        alpha=conf.getDouble("alpha",0.5);
        itemFactors.init(0,0.005);
        W=new DenseMatrix(numItems,numFactors);
        W.init(0,0.005);
        itemBias=new DenseVector(numItems);
        itemBias.init(0,0.1);
        cacheSpec = conf.get("guava.cache.spec", "maximumSize=200,expireAfterAccess=2m");
        userItemsCache = trainMatrix.rowColumnsCache(cacheSpec);
        //itemUsersCache = trainMatrix.columnRowsCache(cacheSpec);
        itemSimilarityMatrix=context.getSimilarity().getSimilarityMatrix();
        for(int itemIdx=0;itemIdx<numItems;itemIdx++)
        {
            for(int anotherItemIdx=itemIdx+1;anotherItemIdx<numItems;anotherItemIdx++)
            {
                if(itemSimilarityMatrix.contains(itemIdx,anotherItemIdx))
                {
                    double sim=itemSimilarityMatrix.get(itemIdx,anotherItemIdx);
                    //double newsim=(sim+1.0)/2;
                    itemSimilarityMatrix.set(itemIdx,anotherItemIdx,sim);
                }

            }
        }

    }

    @Override
    protected void trainModel() throws LibrecException
    {
        for (int iter = 0; iter <numIterations; iter++)
        {
            loss=0;

            DenseMatrix temItemFactors=new DenseMatrix(numItems,numFactors);
            DenseMatrix temWFactors=new DenseMatrix(numItems,numFactors);

            // randomly draw (userIdx, posItemIdx, negItemIdx)
            for (int sampleCount = 0; sampleCount <numUsers; sampleCount++)
            {
                // 抽样 (u,i,j)
                int userIdx=0,posItemIdx=0,negItemIdx=0;
                List<Integer> itemList=null;
                while(true)
                {
                    userIdx= Randoms.uniform(numUsers);
                    try
                    {
                         itemList=userItemsCache.get(userIdx);
                        if(itemList.size()==0)
                        {
                            continue;
                        }
                        posItemIdx=itemList.get(Randoms.uniform(itemList.size()));

                        do
                        {
                            negItemIdx=Randoms.uniform(numItems);
                        }while (itemList.contains(negItemIdx));

                        break;
                    } catch (ExecutionException e)
                    {
                        e.printStackTrace();
                    }
                }
                double posWeight=Math.pow(itemList.size()-1,-alpha);
                double negWeight=Math.pow(itemList.size(),-alpha);

                double posSumsim=0;
                double negSumsim=0;
                double[] vi=new double[numFactors];
                double[] hehe=new double[numFactors];
                //vj的梯度
                double[] vj=new double[numFactors];
                // wi'的梯度
                double[] wi_2=new double[numFactors];
                for(int i_2:itemList)
                {

                    if(i_2!=posItemIdx)
                    {
                        posSumsim+=posWeight*(1-lamdaS+lamdaS*itemSimilarityMatrix.get(posItemIdx,i_2)*DenseMatrix.rowMult(W,i_2,itemFactors,posItemIdx));

                        // vi的梯度
                        for (int factorIdx = 0; factorIdx <numFactors ; factorIdx++)
                        {
                            vi[factorIdx]+=posWeight*(1-lamdaS+lamdaS*itemSimilarityMatrix.get(posItemIdx,i_2))*W.get(i_2,factorIdx);
                        }

                        for (int factorIdx = 0; factorIdx <numFactors ; factorIdx++)
                        {
                            hehe[factorIdx]+=posWeight*((1-lamdaS)+lamdaS*itemSimilarityMatrix.get(posItemIdx,i_2))*itemFactors.get(posItemIdx,factorIdx);
                        }
                    }


                    negSumsim+=negWeight*(1-lamdaS+lamdaS*itemSimilarityMatrix.get(negItemIdx,i_2)*DenseMatrix.rowMult(W,i_2,itemFactors,negItemIdx));


                    for (int factorIdx = 0; factorIdx <numFactors ; factorIdx++)
                    {
                        vj[factorIdx]+=-negWeight*(1-lamdaS+lamdaS*itemSimilarityMatrix.get(negItemIdx,i_2))*W.get(i_2,factorIdx);
                    }



                    for (int factorIdx = 0; factorIdx <numFactors ; factorIdx++)
                    {
                        wi_2[factorIdx]+=-negWeight*((1-lamdaS)+lamdaS*itemSimilarityMatrix.get(negItemIdx,i_2))*itemFactors.get(negItemIdx,factorIdx)+hehe[factorIdx];
                    }



                }

                double error=itemBias.get(posItemIdx)+posSumsim-itemBias.get(negItemIdx)-negSumsim;
                double lossValue = -Math.log(Maths.logistic(error));
                loss += lossValue;

                double bi=itemBias.get(posItemIdx);
                double bj=itemBias.get(negItemIdx);
                itemBias.add(posItemIdx,-learnRate*(-Maths.logistic(-error)+regUser*bi));
                itemBias.add(negItemIdx,-learnRate*(-Maths.logistic(-error)*(-1)+regUser*bj));
                loss+=regUser*(Math.pow(bi,2)+Math.pow(bj,2));


                for (int factorIdx = 0; factorIdx <numFactors ; factorIdx++)
                {
                    double viFactorValue=itemFactors.get(posItemIdx,factorIdx);
                    double vjFactorValue=itemFactors.get(negItemIdx,factorIdx);
                    loss+=regUser*(Math.pow(viFactorValue,2)+Math.pow(vjFactorValue,2));

                    double deltaVi=-Maths.logistic(-error)*vi[factorIdx]+regUser*viFactorValue;
                    double deltaVj=-Maths.logistic(-error)*vj[factorIdx]+regUser*vjFactorValue;

                    temItemFactors.add(posItemIdx,factorIdx,deltaVi);
                    temItemFactors.add(negItemIdx,factorIdx,deltaVj);

                    double wFactorValue=W.get(posItemIdx,factorIdx);
                    double deltaWi=-Maths.logistic(-error)*(-1)*negWeight*((1-lamdaS)+lamdaS*itemSimilarityMatrix.get(negItemIdx,posItemIdx))
                            *itemFactors.get(negItemIdx,factorIdx)+regUser*wFactorValue;
                    temWFactors.add(posItemIdx,factorIdx,deltaWi);

                    loss+=regUser*(Math.pow(viFactorValue,2)+Math.pow(vjFactorValue,2)+Math.pow(wFactorValue,2));

                }

                for(int i_2:itemList)
                {
                    if(i_2!=posItemIdx)
                    {
                        for (int factorIdx = 0; factorIdx <numFactors ; factorIdx++)
                        {
                            double deltaWi_2=-Maths.logistic(-error)*wi_2[factorIdx];
                            temWFactors.add(i_2,factorIdx,deltaWi_2);

                        }
                    }

                }
            }

            itemFactors.addEqual(temItemFactors.scale(-learnRate));
            W.addEqual(temWFactors.scale(-learnRate));

            loss*=0.5;
            if (isConverged(iter) && earlyStop) {
                break;
            }
            //updateLRate(iter);

        }

    }

    @Override
    protected double predict(int userIdx, int itemIdx) throws LibrecException
    {
       double predictRating=itemBias.get(itemIdx);
       List<Integer> itemList=null;
        try
        {
            itemList=userItemsCache.get(userIdx);
        } catch (ExecutionException e)
        {
            e.printStackTrace();
        }
        double posSumsim=0;
        double posWeight=Math.pow(itemList.size()-1,-alpha);
        for(int i_2:itemList)
        {
            if(i_2!=itemIdx)
            {
                posSumsim+=posWeight*(1-lamdaS+lamdaS*itemSimilarityMatrix.get(itemIdx,i_2)*DenseMatrix.rowMult(W,i_2,itemFactors,itemIdx));
            }
        }


        return predictRating+posSumsim;
    }


    public static void main(String[] args) throws LibrecException
    {
        Configuration conf=new Configuration();
        conf.set("data.model.splitter","testset");
        conf.set("dfs.data.dir","D:/mylibrec");
        conf.set("data.input.path","ml100kcopy1/");
        conf.set("data.testset.path","ml100kcopy1/ML100K-copy1-test.txt");

        conf.set("rec.iterator.learnrate", "0.01");
        conf.set("rec.iterator.learnrate.maximum", "0.01");
        conf.set("rec.iterator.maximum", "10");

        conf.set("rec.user.regularization", "0.01");
        conf.set("rec.item.regularization", "0.01");
        conf.set("rec.learnrate.bolddriver","false");
        conf.set("rec.learnrate.decay","1.0");
        conf.set("rec.recommender.earlystop","false");
        conf.set("rec.recommender.verbose","true");
        conf.set("rec.factor.number","20");
        conf.set("rec.recommender.isranking", "true");

        // specific conf
        conf.set("rec.recommender.similarity.key","item");
        conf.set("rec.similarity.class","bcos");
        conf.set("rec.similarity.shrinkage","0");

        conf.set("lamdaS","0.6");
        conf.set("alpha","0.5");
        Randoms.seed(1);


        TextDataModel dataModel = new TextDataModel(conf);
        dataModel.buildDataModel();

        // build recommender context
        RecommenderContext context = new RecommenderContext(conf, dataModel);


        RecommenderSimilarity similarity = new BinaryCosineSimilarity();
        similarity.buildSimilarityMatrix(dataModel);
        context.setSimilarity(similarity);

        //run algorithm
        PFMSM recommender=new PFMSM();
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
