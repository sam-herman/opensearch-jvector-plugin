package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
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
import org.opensearch.knn.index.codec.jvector.datasets.DataSet;
import org.opensearch.knn.index.codec.jvector.datasets.DownloadHelper;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.opensearch.knn.index.codec.jvector.BenchmarkCommon.*;

/************************************************************
 * This benchmark tests the performance of the JVector and Lucene codecs in the plugin
 * with known datasets.
 * Note: Keep in mind that this benchmark is not meant to reproduce the already existing benchmarks of either Lucene or JVector.
 * But rather it is more meant as a qualitative analysis of the relative performance of the codecs in the plugin for certain scenarios.
 ************************************************************/
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Fork(1)
public class FormatBenchmarkWithKnownDatasets {
    Logger log = LogManager.getLogger(FormatBenchmarkWithKnownDatasets.class);
    // large embeddings calculated by Neighborhood Watch.  100k files by default; 1M also available
    private static List<String> LARGE_DATASETS = List.of(
            "ada002-100k",
            "cohere-english-v3-100k",
            "openai-v3-small-100k",
            "nv-qa-v4-100k",
            "colbert-1M",
            "gecko-100k");

    @Param("ada002-100k")
    private String datasetName;
    @Param({JVECTOR_NOT_QUANTIZED/*, JVECTOR_QUANTIZED*/, LUCENE101 })  // This will run the benchmark each codec type
    private String codecType;
    private DataSet dataset;
    private static final int K = 100;

    private Directory directory;
    private float[] queryVector;
    private float expectedMinScoreInTopK;
    private VectorSimilarityFunction vectorSimilarityFunction;
    private double totalRecall = 0.0;
    private int recallCount = 0;

    @Setup
    public void setup() throws IOException {
        // Download datasets
        var mfd = DownloadHelper.maybeDownloadFvecs(datasetName);
        try {
            dataset = mfd.load();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        final Path indexPath = Files.createTempDirectory("jvector-benchmark");
        log.info("Index path: {}", indexPath);
        directory = new NIOFSDirectory(indexPath, FSLockFactory.getDefault());

        // Create index with JVectorFormat
        IndexWriterConfig indexWriterConfig = new IndexWriterConfig();
        indexWriterConfig.setCodec(BenchmarkCommon.getCodec(codecType));
        indexWriterConfig.setUseCompoundFile(true);
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy(true));

        float[][] vectors = new float[dataset.baseVectors.size()][dataset.getDimension()];
        for (int i = 0; i < dataset.baseVectors.size(); i++) {
            vectors[i] = (float[]) dataset.baseVectors.get(i).get();
        }

        // Convert from jVector similarity function to Lucene similarity function
        vectorSimilarityFunction = switch (dataset.similarityFunction) {
            case COSINE -> VectorSimilarityFunction.COSINE;
            case DOT_PRODUCT -> VectorSimilarityFunction.DOT_PRODUCT;
            case EUCLIDEAN -> VectorSimilarityFunction.EUCLIDEAN;
            default -> throw new IllegalStateException("Unexpected similarity function: " + dataset.similarityFunction);
        };
        log.info("Using similarity function: {}", vectorSimilarityFunction);

        try (IndexWriter writer = new IndexWriter(directory, indexWriterConfig)) {
            for (int i = 0; i < vectors.length; i++) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD_NAME, vectors[i], vectorSimilarityFunction));
                writer.addDocument(doc);
            }
            writer.commit();
            log.info("Flushing docs to make them discoverable on the file system and force merging all segments to get a single segment");
            writer.forceMerge(1);
        }

        queryVector = (float[]) dataset.queryVectors.getFirst().get();
        expectedMinScoreInTopK = findExpectedKthMaxScore(queryVector, vectors, vectorSimilarityFunction, K);
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
    public RecallResult benchmarkSearch() throws IOException {
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
}
