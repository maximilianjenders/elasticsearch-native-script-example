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
import redis.clients.jedis.Jedis;

/**
 * Script that scores documents with a language model similarity with linear
 * interpolation, see Manning et al., "Information Retrieval", Chapter 12,
 * Equation 12.12 (link: http://nlp.stanford.edu/IR-book/) This implementation
 * only scores a list of terms on one field.
 */
public class QueryLikelihoodModelScript extends AbstractSearchScript {

    // the field containing the terms that should be scored, must be initialized
    // in constructor from parameters.
    String field;
    // name of the field that holds the word count of a field, see
    // http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/mapping-core-types.html)
    String docLengthField;
    // terms that are used for scoring
    ArrayList<String> terms;
    // lambda parameter
    float lambda;
    // verbose
    boolean verbose = false;
    // upper limit for field length, not mandatory
    int maxFieldLength = 0;

    // redis connection
    Jedis jedis = null;

    final static public String SCRIPT_NAME = "qle_model_script_score";

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
            return new QueryLikelihoodModelScript(params);
        }
    }

    /**
     * @param params
     *            terms that a scored are placed in this parameter. Initialize
     *            them here.
     */
    private QueryLikelihoodModelScript(Map<String, Object> params) {
        params.entrySet();
        // get the terms
        terms = (ArrayList<String>) params.get("terms");
        // get the field
        field = (String) params.get("field");
        // get the field holding the document length
        docLengthField = (String) params.get("word_count_field");

        if (field == null || terms == null || docLengthField == null || !params.containsKey("lambda")) {
            throw new ScriptException("cannot initialize " + SCRIPT_NAME + ": field, terms, length field or lambda parameter missing!");
        }

        // get lambda
        lambda = ((Double) params.get("lambda")).floatValue();

        if (params.containsKey("max_field_length")) {
            maxFieldLength = (int) params.get("max_field_length");
        }

        if (params.containsKey("verbose")) verbose = (boolean) params.get("verbose");

        // by default redis needs to run on localhost
        if (verbose) System.out.println("Connecting to redis at localhost ...");
        jedis = new Jedis("localhost");
        if (verbose) System.out.println("Connecting to redis at localhost ... done.");
        if (verbose) System.out.println("Parameters... field: " + field + " docLengthField: " + docLengthField + " lambda: " + lambda);
    }

    @Override
    public Object run() {
        try {
            double score = 0.0;
            // first, get the ShardTerms object for the field.
            IndexField indexField = indexLookup().get(field);

            boolean atLeastOne = false;

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
                for (String term : this.terms) {
                    // Now, get the ShardTerm object that can be used to access
                    // all
                    // the term statistics
                    IndexFieldTerm indexFieldTerm = indexField.get(term);

                    /*
                     * Collection probability from redis
                     */
                    double tf = this.getTf(term);
                    double M_c = tf;

                    if (!atLeastOne && indexFieldTerm.tf() > 0) atLeastOne = true;
                    /*
                     * Compute M_d, see Manning et al., "Information Retrieval",
                     * Chapter 12, Equation just before Equation 12.9 (link:
                     * http://nlp.stanford.edu/IR-book/)
                     */
                    double M_d = (double) indexFieldTerm.tf() / (double) L_d;
                    /*
                     * compute score contribution for this term, but sum the log
                     * to avoid underflow, see Manning et al.,
                     * "Information Retrieval", Chapter 12, Equation 12.12
                     * (link: http://nlp.stanford.edu/IR-book/)
                     */
                    score += Math.log((1.0 - lambda) * M_c + lambda * M_d);

                    if (verbose) System.out.println("tf: " + tf + " L_d: " + L_d + " M_c: " + M_c + " M_d: " + M_d + " score: " + score);
                }
                if (!atLeastOne || score == 0.0) score = -10000.0;
                if (verbose) System.out.println("QueryLikelihoodScore: " + score);

                return score;
            } else {
                throw new ScriptException("Could not compute language model score, word count field missing.");
            }

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
