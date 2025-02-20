/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.backward_codecs.lucene100.Lucene100Codec;
import org.apache.lucene.backward_codecs.lucene912.Lucene912Codec;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSLockFactory;
import org.apache.lucene.store.NIOFSDirectory;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static org.opensearch.knn.index.codec.jvector.JVectorFormat.DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION;

@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Fork(1)
public class JVectorFormatBenchmark {
    private static final Logger log = LogManager.getLogger(JVectorFormatBenchmark.class);
    private static final String JVECTOR_NOT_QUANTIZED = "jvector_not_quantized";
    private static final String JVECTOR_QUANTIZED = "jvector_quantized";
    private static final String LUCENE101 = "Lucene101";
    private static final String FIELD_NAME = "vector_field";
    private static final int DIMENSION = 128;
    private static final int NUM_DOCS = 10_000;
    private static final int K = 100;
    private static final VectorSimilarityFunction SIMILARITY_FUNCTION = VectorSimilarityFunction.EUCLIDEAN;
    @Param({JVECTOR_NOT_QUANTIZED, JVECTOR_QUANTIZED, LUCENE101 })  // This will run the benchmark each codec type
    private String codecType;

    private Directory directory;
    private double totalRecall = 0.0;
    private int recallCount = 0;

    private Codec getCodec() {
        return switch (codecType) {
            case JVECTOR_NOT_QUANTIZED -> new JVectorCodec(Integer.MAX_VALUE);
            case JVECTOR_QUANTIZED -> new JVectorCodec(DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION);
            case LUCENE101 -> new Lucene101Codec();
            default -> throw new IllegalStateException("Unexpected codec type: " + codecType);
        };
    }

    public static class TestData {
        public static final float[][] VECTORS = new float[NUM_DOCS][DIMENSION];
        public static final float[] QUERY_VECTOR = new float[DIMENSION];
        public static final float EXPECTED_MIN_SCORE_IN_TOP_K;

        static {
            Random random = new Random(42);
            for (int i = 0; i < NUM_DOCS; i++) {
                for (int j = 0; j < DIMENSION; j++) {
                    VECTORS[i][j] = random.nextFloat();
                }
            }

            for (int i = 0; i < DIMENSION; i++) {
                QUERY_VECTOR[i] = random.nextFloat();
            }

            EXPECTED_MIN_SCORE_IN_TOP_K = findExpectedKthMaxScore(QUERY_VECTOR, VECTORS, SIMILARITY_FUNCTION, K);
        }
    }
    @Setup
    public void setup() throws IOException {
        final Path indexPath = Files.createTempDirectory("jvector-benchmark");
        log.info("Index path: {}", indexPath);
        directory = new NIOFSDirectory(indexPath, FSLockFactory.getDefault());

        // Create index with JVectorFormat
        IndexWriterConfig indexWriterConfig = new IndexWriterConfig();
        indexWriterConfig.setCodec(getCodec());
        indexWriterConfig.setUseCompoundFile(true);
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy(true));

        try (IndexWriter writer = new IndexWriter(directory, indexWriterConfig)) {
            for (int i = 0; i < NUM_DOCS; i++) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD_NAME, TestData.VECTORS[i]));
                writer.addDocument(doc);
            }
            writer.commit();
            log.info("Flushing docs to make them discoverable on the file system and force merging all segments to get a single segment");
            writer.forceMerge(1);
        }
    }

    @TearDown
    public void tearDown() throws IOException {
        directory.close();
    }

    // Print average recall after each iteration
    @TearDown(Level.Iteration)
    public void printIterationStats() {
        log.info("Average recall: {}", totalRecall / recallCount);
    }

    @TearDown(Level.Trial)
    public void printFinalStats() {
        log.info("=== Benchmark Results ===");
        log.info("Total Iterations: {}", recallCount);
        log.info("Average Recall: {}", totalRecall / recallCount);
        log.info("=====================");
    }

    @Benchmark
    public RecallResult benchmarkSearch(Blackhole blackhole) throws IOException {
        try (DirectoryReader reader = DirectoryReader.open(directory)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            KnnFloatVectorQuery query = new KnnFloatVectorQuery(FIELD_NAME, TestData.QUERY_VECTOR, K);
            TopDocs topDocs = searcher.search(query, K);

            // Calculate recall
            float recall = calculateRecall(topDocs, TestData.EXPECTED_MIN_SCORE_IN_TOP_K);
            totalRecall += recall;
            recallCount++;
            return new RecallResult(recall);
        }
    }

    private static float calculateRecall(TopDocs topDocs, float expectedMinScoreInTopK) {
        int relevantDocsFound = 0;
        for (int i = 0; i < topDocs.scoreDocs.length; i++) {
            if (topDocs.scoreDocs[i].score >= expectedMinScoreInTopK) {
                relevantDocsFound++;
            }
        }

        return (float) relevantDocsFound / K;
    }

    private static float findExpectedKthMaxScore(
        float[] queryVector,
        float[][] vectors,
        VectorSimilarityFunction similarityFunction,
        int k
    ) {
        final PriorityQueue<Float> topK = new PriorityQueue<>(k);
        for (int i = 0; i < k; i++) {
            topK.add(Float.NEGATIVE_INFINITY);
        }

        for (float[] vector : vectors) {
            float score = similarityFunction.compare(queryVector, vector);
            if (score > topK.peek()) {
                topK.poll();
                topK.add(score);
            }
        }

        return topK.peek();
    }

    // Create a wrapper class for the result
    public static class RecallResult {
        public final float recall;
        public final long timeNs;

        public RecallResult(float recall) {
            this.recall = recall;
            this.timeNs = System.nanoTime();
        }
    }
}
