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
 * Script that scores documents as sum_t(tf_t * (#docs+2)/(df_t+1)), which
 * equals ntn in SMART notation, see Manning et al., "Information Retrieval",
 * Chapter 6, Figure 6.15 (link: http://nlp.stanford.edu/IR-book/) This
 * implementation only scores a list of terms on one field.
 */
public class TempSumBM25ScoreScript extends AbstractSearchScript {

    // defaults according to lucene
    private float DEFAULT_B = 0.75f;
    private float DEFAULT_K1 = 1.2f;

    // the field containing the terms that should be scored, must be initialized
    // in constructor from parameters.
    String field = null;
    // terms that are used for scoring
    ArrayList<String> terms = null;
    // name of the field that holds the word count of a field, see
    // http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/mapping-core-types.html)
    String docLengthField = null;
    // average length of the docLengthField over all documents
    int averageDocLength = 0;

    // redis connection
    Jedis jedis = null;

    // params to Okapi BM25, cf. http://en.wikipedia.org/wiki/Okapi_BM25
    float k1 = 0.0f;
    float b = 0.0f;

    // verbose
    boolean verbose = false;

    final static public String SCRIPT_NAME = "temp_sum_bm25_script_score";

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
            return new TempSumBM25ScoreScript(params);
        }
    }

    /**
     * @param params
     *            terms that a scored are placed in this parameter. Initialize
     *            them here.
     */
    private TempSumBM25ScoreScript(Map<String, Object> params) {
        params.entrySet();
        // get the terms
        terms = (ArrayList<String>) params.get("terms");
        // get the field
        field = (String) params.get("field");
        // word count field
        docLengthField = (String) params.get("word_count_field");
        // average doc length
        if (params.containsKey("word_count_average")) averageDocLength = (int) params.get("word_count_average");

        if (field == null || terms == null || docLengthField == null || averageDocLength <= 0) {
            throw new ScriptException("cannot initialize " + SCRIPT_NAME + ": field or terms, word count field or word count avg parameter missing!");
        }

        // bm25 params
        if (params.containsKey("b")) {
            b = (float) params.get("b");
        } else {
            b = DEFAULT_B;
        }
        if (params.containsKey("k1")) {
            k1 = (float) params.get("k1");
        } else {
            k1 = DEFAULT_K1;
        }

        if (params.containsKey("verbose")) verbose = (boolean) params.get("verbose");

        // by default redis needs to run on localhost
        if (verbose) System.out.println("Connecting to redis at localhost ...");
        jedis = new Jedis("localhost");
        if (verbose) System.out.println("Connecting to redis at localhost ... done.");
    }

    @Override
    public Object run() {
        try {
            float score = 0;
            // first, get the IndexField object for the field.
            IndexField indexField = indexLookup().get(field);

            ScriptDocValues docValues = (ScriptDocValues) doc().get(docLengthField);
            if (docValues == null || !docValues.isEmpty()) {
                long docLength = ((ScriptDocValues.Longs) docValues).getValue();
                float relativeDocLength = ((float) docLength) / ((float) averageDocLength);
                for (String term : terms) {
                    // Now, get the IndexFieldTerm object that can be used to access all
                    // the term statistics
                    IndexFieldTerm indexFieldTerm = indexField.get(term);
                    // compute the most naive tfidf and add to current score
                    int tf = indexFieldTerm.tf();
                    float idf = this.getIdf(term);
                    if (tf != 0) {
                        score += idf * ((float) tf * (k1 + 1)) / ((float) tf + (k1 * (1 - b + (b * relativeDocLength))));
                    }
                    if (verbose) {
                        System.out.println("term: " + term + " idf: " + idf + " tf: " + tf + " k1: " + k1 + " b: " + b + " docLength: " + docLength + " averageDocLength: " + averageDocLength);
                    }
                }
            } else {
                throw new ScriptException("Could not compute temp sum bm25 score, word count field missing.");
            }
            if (verbose)
                System.out.println("TempSumBM25Score: " + score);
            return score;
        } catch (IOException ex) {
            throw new ScriptException("Could not compute temp sum bm25 score: ", ex);
        }
    }

    private String buildIdfKey(String term) {
        return "idf:" + term.replace(":", "-").replace("\"", "'");
    }

    private float getIdf(String term) {
        String value = jedis.get(this.buildIdfKey(term));

        // if the term is new, return the maximum idf
        if (value == null) {
            value = this.jedis.get(this.buildIdfKey("max_idf"));
        }

        return Float.parseFloat(value);
    }
}
