/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.training;

import org.opensearch.action.search.SearchResponse;
import org.opensearch.search.SearchHit;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Transfers float vectors from JVM to native memory.
 */
public class FloatTrainingDataConsumer extends TrainingDataConsumer {
    /**
     * Constructor
     *
     */
    public FloatTrainingDataConsumer() {}

    @Override
    public void accept(List<?> floats) {
        throw new UnsupportedOperationException("Unsupported operation with native memory");
    }

    @Override
    public void processTrainingVectors(SearchResponse searchResponse, int vectorsToAdd, String fieldName) {
        SearchHit[] hits = searchResponse.getHits().getHits();
        List<Float[]> vectors = new ArrayList<>();
        String[] fieldPath = fieldName.split("\\.");

        for (int vector = 0; vector < vectorsToAdd; vector++) {
            Object fieldValue = extractFieldValue(hits[vector], fieldPath);
            if (!(fieldValue instanceof List<?>)) {
                continue;
            }

            List<Number> fieldList = (List<Number>) fieldValue;
            vectors.add(fieldList.stream().map(Number::floatValue).toArray(Float[]::new));
        }

        setTotalVectorsCountAdded(getTotalVectorsCountAdded() + vectors.size());

        accept(vectors);
    }

    private List<byte[]> quantizeVectors(List<?> vectors) throws IOException {
        throw new UnsupportedOperationException("Unsupported operation with native memory");
    }
}
