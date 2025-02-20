/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
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

@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Fork(1)
public class JVectorFormatBenchmark {
    private static final Logger log = LogManager.getLogger(JVectorFormatBenchmark.class);
    private static final String FIELD_NAME = "vector_field";
    private static final int DIMENSION = 128;
    private static final int NUM_DOCS = 10_000;
    private static final int K = 100;
    private static final VectorSimilarityFunction SIMILARITY_FUNCTION = VectorSimilarityFunction.EUCLIDEAN;
    @Param({ "jvector", "lucene" })  // This will run the benchmark for both codecs
    private String codecType;

    private Directory directory;
    private float[][] vectors;
    private float[] queryVector;
    private float expectedMinScoreInTopK;
    private Random random;
    private double totalRecall = 0.0;
    private int recallCount = 0;

    private Codec getCodec() {
        return switch (codecType) {
            case "jvector" -> new JVectorCodec();
            case "lucene" -> new Lucene101Codec();
            default -> throw new IllegalStateException("Unexpected codec type: " + codecType);
        };
    }

    @Setup
    public void setup() throws IOException {
        random = new Random(42);
        final Path indexPath = Files.createTempDirectory("jvector-benchmark");
        log.info("Index path: {}", indexPath);
        // Generate random vectors for indexing
        vectors = new float[NUM_DOCS][DIMENSION];
        for (int i = 0; i < NUM_DOCS; i++) {
            for (int j = 0; j < DIMENSION; j++) {
                vectors[i][j] = random.nextFloat();
            }
        }

        // Generate query vector
        queryVector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            queryVector[i] = random.nextFloat();
        }

        directory = new NIOFSDirectory(indexPath, FSLockFactory.getDefault());

        // Create index with JVectorFormat
        IndexWriterConfig indexWriterConfig = new IndexWriterConfig();
        indexWriterConfig.setCodec(getCodec());
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy());

        try (IndexWriter writer = new IndexWriter(directory, indexWriterConfig)) {
            for (int i = 0; i < NUM_DOCS; i++) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD_NAME, vectors[i]));
                writer.addDocument(doc);
            }
            writer.commit();
            log.info("Flushing docs to make them discoverable on the file system");
            writer.forceMerge(1);
        }

        expectedMinScoreInTopK = findExpectedKthMaxScore(queryVector, vectors, SIMILARITY_FUNCTION, K);
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
            KnnFloatVectorQuery query = new KnnFloatVectorQuery(FIELD_NAME, queryVector, K);
            TopDocs topDocs = searcher.search(query, K);

            // Calculate recall
            float recall = calculateRecall(topDocs, expectedMinScoreInTopK);
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
