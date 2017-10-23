import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.convertor.TextDataConvertor;
import net.librec.data.model.TextDataModel;
import net.librec.data.splitter.RatioDataSplitter;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.ranking.AUCEvaluator;
import net.librec.eval.ranking.NormalizedDCGEvaluator;
import net.librec.eval.ranking.PrecisionEvaluator;
import net.librec.eval.ranking.RecallEvaluator;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.cf.ranking.BPRRecommender;
import net.librec.similarity.PCCSimilarity;
import net.librec.similarity.RecommenderSimilarity;

import java.io.IOException;

public class BPR {
    public static void main(String[] args) throws LibrecException {
    Configuration conf = new Configuration();
    conf.set("dfs.data.dir","D:/librec-2.0.0/data");
    conf.set("data.input.path","filmtrust/rating");
    conf.set("rec.iterator.learnrate", "0.01");
    conf.set("rec.recommender.similarity.key" ,"item");
    conf.set("rec.iterator.learnrate.maximum", "0.01");
    conf.set("rec.iterator.maximum", "10");
    conf.set("rec.user.regularization", "0.01");
    conf.set("rec.item.regularization", "0.01");
    conf.set("rec.factor.number", "30");
    conf.set("rec.recommender.isranking", "true");
    conf.set("data.splitter.ratio", "rating");
    conf.set("data.splitter.trainset.ratio", "0.8");

        TextDataModel dataModel = new TextDataModel(conf);
        dataModel.buildDataModel();

    // build recommender context
    RecommenderContext context = new RecommenderContext(conf, dataModel);

    // build similarity
        // 不需要协同过滤局部相似度 就不要 buildSimilarityMatrix
   /* RecommenderSimilarity similarity = new PCCSimilarity();
    similarity.buildSimilarityMatrix(dataModel);
    context.setSimilarity(similarity);*/

    //run algorithm
    BPRRecommender recommender=new BPRRecommender();
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
