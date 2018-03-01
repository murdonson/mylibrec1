import com.google.common.cache.LoadingCache;
import com.google.common.collect.HashBasedTable;
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
import net.librec.math.structure.MatrixEntry;
import net.librec.math.structure.SparseMatrix;
import net.librec.math.structure.SymmMatrix;
import net.librec.recommender.MatrixFactorizationRecommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.SocialRecommender;
import net.librec.recommender.context.rating.SocialMFRecommender;
import net.librec.similarity.BinaryCosineSimilarity;
import net.librec.similarity.JaccardSimilarity;
import net.librec.similarity.RecommenderSimilarity;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;

public class USG extends SocialRecommender
{

    private SymmMatrix symmMatrix;
    private Double eta;
    protected static String cacheSpec;
    // 用户关注的朋友
    protected LoadingCache<Integer, List<Integer>> socialFreindsCache;


    protected LoadingCache<Integer, List<Integer>> locationCache;

    private Table<Integer,Float,Float> poitable;

    private double a;
    private double b;


    private double u_cij;
    private double s_cij;
    private double g_cij;

    private int[] distribution=new int[20000];
    @Override
    public void setup() throws LibrecException
    {
        super.setup();
        eta=0.5;



        // 用于用户协同过滤部分
        symmMatrix=context.getSimilarity().getSimilarityMatrix();
        cacheSpec = conf.get("guava.cache.spec", "maximumSize=200,expireAfterAccess=2m");
        socialFreindsCache=socialMatrix.rowColumnsCache(cacheSpec);
        locationCache=trainMatrix.rowColumnsCache(cacheSpec);



        // 地理部分 读取 poi 信息  地点ID 经度 维度
        poitable =HashBasedTable.create();
        try
        {
            FileInputStream fin=new FileInputStream("D:\\code\\mylibrec\\Gowalla\\poi\\Gowalla_poi_coos.txt");
            InputStreamReader inReader=new InputStreamReader(fin);
            BufferedReader bufReader=new BufferedReader(inReader);
            String line=null;
            while((line=bufReader.readLine())!=null)
            {
                String[] s=line.trim().split("[ \t,]+");
                int locId=Integer.valueOf(s[0]);
                float lat=Float.valueOf(s[1]);
                float lng=Float.valueOf(s[2]);
                poitable.put(locId,lat,lng);
            }

        } catch (Exception e)
        {
            e.printStackTrace();
        }


        // 社交部分
        JaccardSimilarity jaccardSimilarity=new JaccardSimilarity();
        //  eta*共同的朋友 +（1-eta）*共同旅游过的地点(等同于共同买过的商品)
        for(MatrixEntry entry:socialMatrix)
        {
            int userIdx=entry.row();
            int trusteeIdx=entry.column();

           double sim1= jaccardSimilarity.getCorrelation(trainMatrix.row(userIdx),trainMatrix.row(trusteeIdx));
           double sim2=jaccardSimilarity.getCorrelation(socialMatrix.row(userIdx),socialMatrix.row(trusteeIdx));
           socialMatrix.set(userIdx,trusteeIdx,eta*sim2+(1-eta)*sim1);
        }


        computeDistance(trainMatrix);

        }


    public  void computeDistance(SparseMatrix trainMatrix)
    {
        int hehe=0;
        for(MatrixEntry entry:trainMatrix)
        {
            if((hehe++)%100==0)
            {
                System.out.println(hehe);
            }
            int userIdx=entry.row();
            try
            {
                // 用户访问的地点列表
                List<Integer> list=locationCache.get(userIdx);
                Object[] loc=list.toArray();


                // 选两个不同的地点 提取出经纬度 然后根据公式计算相似度
                for(int i=0;i<loc.length;i++)
                {
                    int loc1= (int) loc[i];

                    for(int j=i+1;j<loc.length;j++)
                    {
                        int loc2= (int) loc[j];
                        poitable.cellSet();
                        Map<Float,Float> map1=poitable.row(loc1);
                        Map<Float,Float> map2=poitable.row(loc2);
                        int distance= (int)computePoisimilarity(map1,map2);
                        distribution[distance]++;

                    }
                }

                int total=0;
                for (int i : distribution)
                {
                    total+=i;
                }

                // distribution 数组索引是两个poi地点之间的距离 值是check in 概率
                for (int i = 0; i < distribution.length; i++)
                {
                    distribution[i]/=total;
                }

                } catch (ExecutionException e)
            {
                e.printStackTrace();
            }
        }
    }


    private double computePoisimilarity(Map<Float,Float> map1,Map<Float,Float> map2)
    {
        float lat1=0,lat2=0,lng1=0,lng2=0;
         for(Map.Entry<Float,Float> entry:map1.entrySet())
         {
              lat1=entry.getKey();
              lng1=entry.getValue();
         }


        for(Map.Entry<Float,Float> entry:map2.entrySet())
        {
             lat2=entry.getKey();
             lng2=entry.getValue();
        }

        if(Math.abs(lat1-lat2)<Math.exp(-6)&&Math.abs(lng1-lng2)<Math.exp(-6))
        {
            return 0;
        }

         double degreestoRadians=Math.PI/180;
         double phi1=(90.0-lat1)*degreestoRadians;
         double phi2=(90-lat2)*degreestoRadians;
         double theta1=lng1*degreestoRadians;
         double theta2=lng2*degreestoRadians;

         double cos=(Math.sin(phi1)*Math.sin(phi2)*Math.cos(theta1 - theta2) +
                 Math.cos(phi1)*Math.cos(phi2));
         double arc=Math.acos(cos);
         return arc*6371;

    }




    protected void trainModel() throws LibrecException
    {
        System.out.println("trainModel");
        double w0= Randoms.random();
        double w1=Randoms.random();
        double lambda_w=0.1;
        double alpha=0.1;
        double d_w0=0;
        double d_w1=0;

        double[] x=new double[122350];
        double[] t=new double[122350];

        for(int m=0;m<distribution.length;m++)
        {
            x[m]=Math.log10(m);
            t[m]=Math.log10(distribution[m]);
        }


        for(int i=0;i<numIterations;i++)
        {
            loss=0;
            for(int j=0;j<x.length;j++)
            {
                d_w0 += (w0 + w1 * x[j] - t[j]);
                d_w1 += (w0 + w1 * x[j] - t[j]) * x[j];
            }
            w0 -= alpha * (d_w0 + lambda_w * w0);
            w1 -= alpha * (d_w1 + lambda_w * w1);



            for(int k=0;k<x.length;k++)
            {
                loss += 0.5 * Math.pow((w0 + w1 * x[k] - t[k]),2);
            }

            loss += 0.5 * lambda_w * (w0*w0 + w1*w1);

            System.out.println("iter: "+i+" loss: "+loss);
            if (isConverged(i) && earlyStop) {
                break;
            }
        }

       a=Math.pow(10,w0);
       b=w1;



    }








    @Override
    protected double predict(int userIdx, int itemIdx) throws LibrecException
    {

        // 用户协同过滤部分  找所有跟当前用户有共同访问地点的用户
        u_cij=0;
        double total1=0;
        for(int k=0;k<numUsers;k++)
        {
                if(k==userIdx || symmMatrix.get(userIdx,k)==0)
                {
                    continue;
                }
                u_cij+=symmMatrix.get(userIdx,k)*trainMatrix.get(k,itemIdx);
                total1+=symmMatrix.get(userIdx,k);
        }
        u_cij=u_cij/total1;




        // 社交部分   找用户的关注列表用户
        s_cij=0;
        double total2=0;
        try
        {
            List list=socialFreindsCache.get(userIdx);
            Object[] a=list.toArray();
            if(a.length==0)
            {
                s_cij=0;
            }
            else {
                for(int j=0;j<a.length;j++)
                {
                    s_cij+=socialMatrix.get(userIdx, (Integer) a[j])*trainMatrix.get((Integer) a[j],itemIdx);
                    total2+=socialMatrix.get(userIdx, (Integer) a[j]);

                }
                s_cij=s_cij/total2;
            }
        } catch (ExecutionException e)
        {
            e.printStackTrace();
        }


        g_cij=0;
        double total3=0;
        double liancheng=1;

        // 地理部分
        try
        {
            List loclist=locationCache.get(userIdx);
            Object[] o=loclist.toArray();
            if(o.length==0)
            {
                g_cij=0;
            }
            else {
                for(int m=0;m<o.length;m++)
                {
                    Map<Float,Float> map1=poitable.row(itemIdx);
                    Map<Float,Float> map2=poitable.row((Integer) o[m]);
                    double distance= computePoisimilarity(map1,map2);
                    distance=Math.max(0.01,distance);
                    liancheng*=distance;
                }

                g_cij=a*(Math.pow(liancheng,b));
            }


        } catch (ExecutionException e)
        {
            e.printStackTrace();
        }

        // 硬编码
        return (1-0.1-0.1)*u_cij+0.1*s_cij+0.1*g_cij;





    }

    public static void main(String[] args) throws LibrecException
    {
        Configuration conf=new Configuration() ;
        conf.set("data.model.splitter", "testset");
        conf.set("dfs.data.dir", "D:/code/mylibrec");
        conf.set("data.input.path", "Gowalla/rating");
        conf.set("data.testset.path", "Gowalla/rating/Gowalla_test.txt");


        conf.set("data.appender.class","social");
        conf.set("data.appender.path","Gowalla/trust");



        conf.set("rec.iterator.learnrate", "0.01");
        conf.set("rec.recommender.similarity.key" ,"user");
        conf.set("rec.iterator.learnrate.maximum", "0.01");


        conf.set("rec.iterator.maximum", "100");
        conf.set("rec.user.regularization", "0.001");
        conf.set("rec.item.regularization", "0.001");
        conf.set("rec.factor.number", "10");
        conf.set("rec.recommender.isranking", "true");


        /*conf.set("data.splitter.ratio", "rating");
        conf.set("data.splitter.trainset.ratio", "0.8");*/



        TextDataModel dataModel = new TextDataModel(conf);
        dataModel.buildDataModel();
        RecommenderContext context = new RecommenderContext(conf, dataModel);

        RecommenderSimilarity similarity = new BinaryCosineSimilarity();
        similarity.buildSimilarityMatrix(dataModel);
        context.setSimilarity(similarity);


        USG recommender=new USG();
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
