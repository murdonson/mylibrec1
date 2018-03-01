import com.google.common.cache.LoadingCache;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.model.ArffDataModel;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.ranking.AUCEvaluator;
import net.librec.eval.ranking.NormalizedDCGEvaluator;
import net.librec.eval.ranking.PrecisionEvaluator;
import net.librec.eval.ranking.RecallEvaluator;
import net.librec.math.algorithm.Maths;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.SparseVector;
import net.librec.math.structure.TensorEntry;
import net.librec.math.structure.VectorEntry;
import net.librec.recommender.FactorizationMachineRecommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.item.RecommendedItemList;
import net.librec.recommender.item.RecommendedList;

import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutionException;

public class PRFM extends FactorizationMachineRecommender
{

    private String cacheSpec;
    private  double learnRate;
    private LoadingCache<Integer, List<Integer>> userItemsCache;
    @Override
    protected void setup() throws LibrecException
    {
        super.setup();
        learnRate=0.01d;
        int[] numDroppedItemsArray = new int[numUsers]; // for AUCEvaluator
        int maxNumTestItemsByUser = 0; //for idcg
        for (int userIdx = 0; userIdx < 400; ++userIdx) {
            numDroppedItemsArray[userIdx] = numItems - trainTensor.rateMatrix().rowSize(userIdx);
            int numTestItemsByUser = testTensor.rateMatrix().rowSize(userIdx);
            maxNumTestItemsByUser = maxNumTestItemsByUser < numTestItemsByUser ? numTestItemsByUser : maxNumTestItemsByUser;
        }
        conf.setInts("rec.eval.auc.dropped.num", numDroppedItemsArray);
        conf.setInt("rec.eval.item.test.maxnum", maxNumTestItemsByUser);



        cacheSpec = conf.get("guava.cache.spec", "maximumSize=200,expireAfterAccess=2m");
        //userItemsCache = trainMatrix.rowColumnsCache(cacheSpec);
        userItemsCache=trainTensor.rateMatrix().rowColumnsCache(cacheSpec);

    }

    @Deprecated
    protected double predict(int userIdx, int itemIdx) throws LibrecException
    {
        return 0;
    }

    @Override
    protected void trainModel() throws LibrecException
    {
        for (int iter = 0; iter < numIterations; iter++)
        {
            lastLoss=loss;
            loss = 0.0;
            for (int sampleCount = 0; sampleCount <500; sampleCount++)
            {
                // 抽样 (u,i,j)
                int userIdx = 0, posItemIdx = 0, negItemIdx = 0;
                List<Integer> itemList = null;
                while (true)
                {
                    userIdx = Randoms.uniform(numUsers);
                    try
                    {
                        itemList = userItemsCache.get(userIdx);
                        if(itemList.size()==0)
                            continue;
                        posItemIdx = itemList.get(Randoms.uniform(itemList.size()));
                        do
                        {
                            negItemIdx = Randoms.uniform(numItems);
                        } while (itemList.contains(negItemIdx));

                        break;
                    } catch (ExecutionException e)
                    {
                        e.printStackTrace();
                    }
                }

                List<Integer> indexList=trainTensor.getIndices(userIdx,posItemIdx); // 取得 user item对应记录列表
                int index=indexList.get(0);// 我只取第一个  这是第几条记录
                int []posentryKeys=trainTensor.keys(index);

                SparseVector vector = tenserKeysToFeatureVector(posentryKeys);
                double posY=predict(0,0,vector);// 前两个参数随便填

                // 除了物品idx 其他不变
                posentryKeys[1]=negItemIdx;
                vector=tenserKeysToFeatureVector(posentryKeys);
                double negY=predict(0,0,vector);
                double err=posY-negY;
                loss+=-Math.log(Maths.logistic(err));

                double gradLoss=-Maths.logistic(-err);

                loss+=regW0 * w0 * w0;

                double hW0 = 1;
                double gradW0 = gradLoss * hW0 + regW0 * w0;

                // update w0
                w0 += -learnRate * gradW0;


                // 1-way interactions
                for(VectorEntry ve: vector){
                    int l = ve.index();
                    double oldWl = W.get(l);
                    double hWl = ve.get();
                    double gradWl = gradLoss * hWl + regW * oldWl;
                    W.add(l, -learnRate * gradWl);

                    loss += regW * oldWl * oldWl;

                    // 2-way interactions
                    for (int f = 0; f < k; f++) {
                        double oldVlf = V.get(l, f);
                        double hVlf = 0;
                        double xl =ve.get();
                        for(VectorEntry ve2: vector){
                            int j = ve2.index();
                            if(j!=l){
                                hVlf += xl * V.get(j, f) * ve2.get();
                            }
                        }

                        double gradVlf = gradLoss * hVlf + regF * oldVlf;
                        V.add(l, f, -learnRate * gradVlf);
                        loss += regF * oldVlf * oldVlf;
                    }
                }
            }
            loss *= 0.5;
            if (isConverged(iter) && earlyStop) {
                break;
            }

        }
    }

    @Override
    protected RecommendedList recommendRank() throws LibrecException {
        testMatrix = testTensor.rateMatrix();
        recommendedList = new RecommendedItemList(numUsers - 1, numUsers);
        // each user-item pair appears in the final recommend list only once
        Table<Integer, Integer, Double> ratingMapping = HashBasedTable.create();
        // user特征的顺序  userDimension=0  用户第一个特征  itemDimension=1 商品第二个特征

        for (int userIdx = 0; userIdx < numUsers; ++userIdx) {
            Set<Integer> itemSet = trainTensor.rateMatrix().getColumnsSet(userIdx);
            for (int itemIdx = 0; itemIdx < numItems; ++itemIdx) {
                if (itemSet.contains(itemIdx)) {
                    continue;
                }

                // 这个itemIdx是非训练集的
                int []posentryKeys={userIdx,itemIdx,99};
                SparseVector featureVector = tenserKeysToFeatureVector(posentryKeys);
                double predictRating = predict(0, 0, featureVector, true);
                if (Double.isNaN(predictRating)) {
                    continue;
                }
                recommendedList.addUserItemIdx(userIdx, itemIdx, predictRating);
            }
            recommendedList.topNRankItemsByUser(userIdx, topN);
        }

        if(recommendedList.size()==0){
            throw new IndexOutOfBoundsException("No item is recommended, there is something error in the recommendation algorithm! Please check it!");
        }

        return recommendedList;



    }


    private int[] getUserItemIndex(SparseVector x) {
        int[] inds = x.getIndex();

        int userInd = inds[0];
        int itemInd = inds[1] - numUsers;

        return new int[]{userInd, itemInd};
    }






    public static void main(String[] args) throws LibrecException
    {
        Configuration conf=new Configuration();
        conf.set("data.model.splitter","ratio");
        conf.set("dfs.data.dir","D:/librec-2.0.0/data");
        conf.set("data.input.path","test/datamodeltest/ratings.arff");
       conf.set("data.column.format","UIR");
        conf.set(" data.convertor.format","arff");
        conf.set("data.model.format","arff");

        conf.set("learnRate", "0.01");
        conf.set("rec.iterator.learnrate.maximum", "0.01");
        conf.set("rec.iterator.maximum", "20");

        conf.set("rec.user.regularization", "0.01");
        conf.set("rec.item.regularization", "0.01");
        conf.set("rec.learnrate.bolddriver","false");
        conf.set("rec.learnrate.decay","1.0");
        conf.set("rec.recommender.earlystop","false");
        conf.set("rec.recommender.verbose","true");
        conf.set("rec.factor.number","10");
        conf.set("rec.recommender.isranking", "true");


        Randoms.seed(1);


        ArffDataModel dataModel = new ArffDataModel(conf);
        dataModel.buildDataModel();

        // build recommender context
        RecommenderContext context = new RecommenderContext(conf, dataModel);
        //run algorithm
        PRFM recommender=new PRFM();
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
            //auc+=recommender.evaluate(evaluator);
            ndcg+=recommender.evaluate(evaluator1);
            prec+=recommender.evaluate(evaluator2);
            rec+=recommender.evaluate(evaluator3);

        }
        //System.out.println("AUC:" + auc/1);
        System.out.println("NDCG:" + ndcg/1);
        System.out.println("PRECISION:" + prec/1);
        System.out.println("REALL:" + rec/1);

    }







}
