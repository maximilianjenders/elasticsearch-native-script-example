package org.elasticsearch.examples.nativescript.script;

import java.io.IOException;
import java.util.*;

import org.apache.lucene.search.Scorer;
import org.elasticsearch.common.Nullable;
import org.elasticsearch.index.fielddata.ScriptDocValues;
import org.elasticsearch.script.AbstractSearchScript;
import org.elasticsearch.script.ExecutableScript;
import org.elasticsearch.script.NativeScriptFactory;
import org.elasticsearch.script.ScriptException;
import org.elasticsearch.search.lookup.IndexField;
import org.elasticsearch.search.lookup.IndexFieldTerm;
import redis.clients.jedis.Jedis;

/**
 * Script that scores documents with a language model similarity with linear
 * interpolation, see Manning et al., "Information Retrieval", Chapter 12,
 * Equation 12.12 (link: http://nlp.stanford.edu/IR-book/) This implementation
 * only scores a list of terms on one field.
 */
public class KullbackLeiblerScoreScript extends AbstractSearchScript {

    // the field containing the terms that should be scored, must be initialized
    // in constructor from parameters.
    String field = null;
    // name of the field that holds the word count of a field, see
    // http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/mapping-core-types.html)
    String docLengthField = null;
    // terms that are used for scoring
    Map<String, Double> queryModel = null;
    // upper limit for field length, not mandatory
    int maxFieldLength = 0;
    // verbose switch, default false, not mandatory
    boolean verbose = false;
    // lambda parameter
    float lambda;

    // redis connection
    Jedis jedis = null;

    final static public String SCRIPT_NAME = "kullback_leibler_script_score";

    @Override
    public void setScorer(Scorer scorer) {
        // ignore
    }

    /**
     * Factory that is registered in
     * {@link org.elasticsearch.examples.nativescript.plugin.NativeScriptExamplesPlugin#onModule(org.elasticsearch.script.ScriptModule)}
     * method when the plugin is loaded.
     */
    public static class Factory implements NativeScriptFactory {

        /**
         * This method is called for every search on every shard.
         *
         * @param params
         *            list of script parameters passed with the query
         * @return new native script
         */
        @Override
        public ExecutableScript newScript(@Nullable Map<String, Object> params) {
            return new KullbackLeiblerScoreScript(params);
        }
    }

    /**
     * @param params
     *            terms that a scored are placed in this parameter. Initialize
     *            them here.
     */
    private KullbackLeiblerScoreScript(Map<String, Object> params) {
        params.entrySet();
        // get the model
        queryModel = (Map<String, Double>) params.get("query_model");
        // get the field
        field = (String) params.get("field");
        // get the field holding the document length
        docLengthField = (String) params.get("word_count_field");
        if (field == null || queryModel == null || docLengthField == null || !params.containsKey("lambda")) {
            throw new ScriptException("cannot initialize " + SCRIPT_NAME + ": field, query_model, length or lambda field parameter missing!");
        }

        // get lambda
        lambda = ((Double) params.get("lambda")).floatValue();


        if (params.containsKey("max_field_length")) {
            maxFieldLength = (int) params.get("max_field_length");
        }
        if (params.containsKey("verbose")) {
            verbose = (boolean) params.get("verbose");
        }
        // by default redis needs to run on localhost
        if (verbose) System.out.println("Connecting to redis at localhost ...");
        jedis = new Jedis("localhost");
        if (verbose) System.out.println("Connecting to redis at localhost ... done.");
    }

    @Override
    public Object run() {
        try {
            double score = 0.0;
            // first, get the ShardTerms object for the field.
            IndexField indexField = indexLookup().get(field);

            if (this.maxFieldLength > 0) {
                int fieldLength = ((String) source().get(field)).length();
                if (fieldLength > this.maxFieldLength) {
                    if (verbose)
                        System.out.println("too long");
                    return -1000.0;
                }
            }

            /*
             * document length cannot be obtained by the shardTerms, we use the
             * word_count field instead (link:
             * http://www.elasticsearch.org/guide
             * /en/elasticsearch/reference/current/mapping-core-types.html)
             */
            boolean atLeastOne = false;
            ScriptDocValues docValues = (ScriptDocValues) doc().get(docLengthField);
            ScriptDocValues termValues = (ScriptDocValues) doc().get(field);

            if (docValues == null || !docValues.isEmpty() || termValues == null || termValues.isEmpty()) {
                long L_d = ((ScriptDocValues.Longs) docValues).getValue();
                List<String> documentTerms = ((ScriptDocValues.Strings) termValues).getValues();

                HashSet<String> vocabularyTerms = new HashSet<>();
                vocabularyTerms.addAll(queryModel.keySet());
                vocabularyTerms.addAll(documentTerms);

                for (String term : vocabularyTerms) {

                    // Now, get the ShardTerm object that can be used to access
                    // all
                    // the term statistics
                    IndexFieldTerm indexFieldTerm = indexField.get(term);

                    int tf = indexFieldTerm.tf();
                    /*
                     * compute Kullback Leibler Divergence , see Manning et al.,
                     * "Information Retrieval", Chapter 12, Equation 12.10
                     * (link: http://nlp.stanford.edu/IR-book/)
                     */
                    if (!atLeastOne && indexFieldTerm.tf() > 0) atLeastOne = true;

                    double P_t_Mq = queryModel.containsKey(term) ? queryModel.get(term) : 0.0;
                    double P_t_Md = (double) tf / (double)L_d;
                    double P_t_Mc = this.getTf(term);


                    double KL = ((1.0 - lambda) * P_t_Mc + lambda * P_t_Mq) * Math.log((1.0 - lambda) * P_t_Mc + lambda * P_t_Md);
//                    double KL = P_t_Mq * Math.log((1.0 - lambda) * P_t_Mc + lambda * P_t_Md);

                    score += KL;

                    if (verbose)
                        System.out.println("term: " + term + ", tf: " + tf + ", P_t_Mq: " + P_t_Mq + ", P_t_Md: "+ P_t_Md + ", P_t_Mc: "+ P_t_Mc + ", KL: " + KL);

                }
            } else {
                throw new ScriptException("Could not compute language model score, word count field missing or unable to retrieve field terms");
            }
            if (verbose) {
                System.out.println(score);
            }
            if (!atLeastOne)
                score = -1000.0;
            return score;

        } catch (IOException ex) {
            throw new ScriptException("Could not compute language model score: ", ex);
        }
    }


    private String buildTfKey(String term) {
        return "tf:" + term.replace(":", "-").replace("\"", "'");
    }

    private double getTf(String term) {
        String value = jedis.get(this.buildTfKey(term));

        // if the term is new, return the maximum idf
        if (value == null) {
            value = this.jedis.get(this.buildTfKey("min_tf"));
        }

        return Double.parseDouble(value);
    }
}
