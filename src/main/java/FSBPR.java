import com.google.common.cache.LoadingCache;
import net.librec.common.LibrecException;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.DenseVector;
import net.librec.math.structure.SymmMatrix;
import net.librec.recommender.MatrixFactorizationRecommender;
import net.librec.recommender.cf.rating.MFALSRecommender;

import java.util.List;
import java.util.concurrent.ExecutionException;

public class FSBPR extends MatrixFactorizationRecommender
{
    private DenseMatrix W;
    private double lamdaS;
    private double alpha;
    private String cacheSpec;
    private DenseVector itemBias;
    private LoadingCache<Integer, List<Integer>> userItemsCache;
    @Override
    protected void setup() throws LibrecException
    {
        super.setup();
        lamdaS=conf.getDouble("lamdaS",0.5);
        alpha=conf.getDouble("alpha",0.5);
        itemFactors.init(0,0.01);
        W=new DenseMatrix(numItems,numFactors);
        W.init(0,0.01);
        itemBias=new DenseVector(numItems);
        itemBias.init(0,0.1);
        cacheSpec = conf.get("guava.cache.spec", "maximumSize=200,expireAfterAccess=2m");
        userItemsCache = trainMatrix.rowColumnsCache(cacheSpec);
    }

    @Override
    protected void trainModel() throws LibrecException
    {
        for (int iter = 0; iter <numIterations; iter++)
        {
            loss = 0;

            DenseMatrix temItemFactors = new DenseMatrix(numItems, numFactors);
            DenseMatrix temWFactors = new DenseMatrix(numItems, numFactors);

            // randomly draw (userIdx, posItemIdx, negItemIdx)
            for (int sampleCount = 0; sampleCount < numUsers; sampleCount++)
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

                double posWeight=Math.pow(itemList.size()-1,-alpha);
                double negWeight=Math.pow(itemList.size(),-alpha);

                double posSumsim=0;
                double negSumsim=0;


                for(int i_2:itemList)
                {
                    if(i_2!=posItemIdx)
                    {
                        //posSumsim+=(1-lamdaS)
                    }
                }





            }


        }






    }


}
