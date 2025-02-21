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

import static org.opensearch.knn.index.codec.jvector.JVectorFormat.DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION;

/**
 * Benchmark to compare the performance of JVector and Lucene codecs with random vectors.
 * The benchmark generates random vectors and indexes them using JVector and Lucene codecs.
 * It then performs a search using a random query vector and measures the recall.
 * Note: This benchmark is not meant to reproduce the already existing benchmarks of either Lucene or JVector.
 * But rather it is more meant as a qualitative analysis of the relative performance of the codecs in the plugin for certain scenarios.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Fork(1)
public class FormatBenchmarkRandomVectors {
    private static final Logger log = LogManager.getLogger(FormatBenchmarkRandomVectors.class);
    private static final String JVECTOR_NOT_QUANTIZED = "jvector_not_quantized";
    private static final String JVECTOR_QUANTIZED = "jvector_quantized";
    private static final String LUCENE101 = "Lucene101";
    private static final String FIELD_NAME = "vector_field";
    private static final int K = 100;
    private static final VectorSimilarityFunction SIMILARITY_FUNCTION = VectorSimilarityFunction.EUCLIDEAN;
    @Param({JVECTOR_NOT_QUANTIZED/*, JVECTOR_QUANTIZED*/, LUCENE101 })  // This will run the benchmark each codec type
    private String codecType;
    @Param({"1000", "10000", "100000"})
    private int numDocs;
    @Param({"128", "256", "512", "1024"})
    private int dimension;

    private float[][] vectors;
    private float[] queryVector;
    private float expectedMinScoreInTopK;

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

    @Setup
    public void setup() throws IOException {
        vectors = new float[numDocs][dimension];
        queryVector = new float[dimension];
        log.info("Generating {} random vectors of dimension {}", numDocs, dimension);
        // Generate random vectors
        Random random = new Random(42);
        for (int i = 0; i < numDocs; i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = random.nextFloat();
            }
        }

        for (int i = 0; i < dimension; i++) {
            queryVector[i] = random.nextFloat();
        }

        expectedMinScoreInTopK = BenchmarkCommon.findExpectedKthMaxScore(queryVector, vectors, SIMILARITY_FUNCTION, K);

        final Path indexPath = Files.createTempDirectory("jvector-benchmark");
        log.info("Index path: {}", indexPath);
        directory = new NIOFSDirectory(indexPath, FSLockFactory.getDefault());

        // Create index with JVectorFormat
        IndexWriterConfig indexWriterConfig = new IndexWriterConfig();
        indexWriterConfig.setCodec(getCodec());
        indexWriterConfig.setUseCompoundFile(true);
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy(true));

        try (IndexWriter writer = new IndexWriter(directory, indexWriterConfig)) {
            for (int i = 0; i < numDocs; i++) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD_NAME, vectors[i]));
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
    public BenchmarkCommon.RecallResult benchmarkSearch(Blackhole blackhole) throws IOException {
        try (DirectoryReader reader = DirectoryReader.open(directory)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            KnnFloatVectorQuery query = new KnnFloatVectorQuery(FIELD_NAME, queryVector, K);
            TopDocs topDocs = searcher.search(query, K);

            // Calculate recall
            float recall = BenchmarkCommon.calculateRecall(topDocs, expectedMinScoreInTopK);
            totalRecall += recall;
            recallCount++;
            return new BenchmarkCommon.RecallResult(recall);
        }
    }
}
