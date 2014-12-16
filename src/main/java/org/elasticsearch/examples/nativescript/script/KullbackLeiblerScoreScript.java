package org.elasticsearch.examples.nativescript.script;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;

import org.apache.lucene.search.Scorer;
import org.elasticsearch.common.Nullable;
import org.elasticsearch.index.fielddata.ScriptDocValues;
import org.elasticsearch.script.AbstractSearchScript;
import org.elasticsearch.script.ExecutableScript;
import org.elasticsearch.script.NativeScriptFactory;
import org.elasticsearch.script.ScriptException;
import org.elasticsearch.search.lookup.IndexField;
import org.elasticsearch.search.lookup.IndexFieldTerm;

/**
 * Script that scores documents with a language model similarity with linear
 * interpolation, see Manning et al., "Information Retrieval", Chapter 12,
 * Equation 12.12 (link: http://nlp.stanford.edu/IR-book/) This implementation
 * only scores a list of terms on one field.
 */
public class KullbackLeiblerScoreScript extends AbstractSearchScript {

    // the field containing the terms that should be scored, must be initialized
    // in constructor from parameters.
    String field;
    // name of the field that holds the word count of a field, see
    // http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/mapping-core-types.html)
    String docLengthField;
    // terms that are used for scoring
    Map<String, Double> queryModel;
    // upper limit for field length, not mandatory
    int maxFieldLength = -1;
    // verbose switch, default false, not mandatory
    boolean verbose = false;

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
        if (field == null || queryModel == null || docLengthField == null) {
            throw new ScriptException("cannot initialize " + SCRIPT_NAME + ": field, query_model or length field parameter missing!");
        }

        if (params.containsKey("max_field_length")) {
            maxFieldLength = (int) params.get("max_field_length");
        }
        if (params.containsKey("verbose")) {
            verbose = (boolean) params.get("verbose");
        }
    }

    @Override
    public Object run() {
        try {
            double score = 0.0;
            // first, get the ShardTerms object for the field.
            IndexField indexField = indexLookup().get(field);
            long T = indexField.sumttf();

            if (this.maxFieldLength > 0) {
                int fieldLength = ((String) source().get(field)).length();
                if (fieldLength > this.maxFieldLength) {
                    if (verbose)
                        System.out.println("too long");
                    return -100.0;
                }
            }

            /*
             * document length cannot be obtained by the shardTerms, we use the
             * word_count field instead (link:
             * http://www.elasticsearch.org/guide
             * /en/elasticsearch/reference/current/mapping-core-types.html)
             */
            ScriptDocValues docValues = (ScriptDocValues) doc().get(docLengthField);
            if (docValues == null || !docValues.isEmpty()) {
                long L_d = ((ScriptDocValues.Longs) docValues).getValue();
                for (Map.Entry<String, Double> entry : queryModel.entrySet()) {
                    String term = entry.getKey();

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
                    if (tf == 0) {
                        continue;
                    }
                    double P_t_Mq = entry.getValue();
                    double P_t_Md = tf / (double)L_d;
                    double KL = P_t_Mq * Math.log(P_t_Mq / P_t_Md);

                    score += KL;

                    if (verbose)
                        System.out.println("term: " + term + ", tf: " + tf + ", P_t_Mq: " + P_t_Mq + ", P_t_Md: "+ P_t_Md + ", KL: " + KL);

                }
            } else {
                throw new ScriptException("Could not compute language model score, word count field missing.");
            }
            if (verbose) {
                System.out.println(score);
            }
            return score;

        } catch (IOException ex) {
            throw new ScriptException("Could not compute language model score: ", ex);
        }
    }

}
