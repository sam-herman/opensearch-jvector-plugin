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

import java.util.ArrayList;
import java.util.List;

/**
 * Transfers byte vectors from JVM to native memory.
 */
public class ByteTrainingDataConsumer extends TrainingDataConsumer {

    /**
     * Constructor
     *
     */
    public ByteTrainingDataConsumer() {
        super();
    }

    @Override
    public void accept(List<?> byteVectors) {}

    @Override
    public void processTrainingVectors(SearchResponse searchResponse, int vectorsToAdd, String fieldName) {
        SearchHit[] hits = searchResponse.getHits().getHits();
        List<byte[]> vectors = new ArrayList<>();
        String[] fieldPath = fieldName.split("\\.");

        for (int vector = 0; vector < vectorsToAdd; vector++) {
            Object fieldValue = extractFieldValue(hits[vector], fieldPath);

            byte[] byteArray;
            if (!(fieldValue instanceof List<?>)) {
                continue;
            }
            List<Number> fieldList = (List<Number>) fieldValue;
            byteArray = new byte[fieldList.size()];
            for (int i = 0; i < fieldList.size(); i++) {
                byteArray[i] = fieldList.get(i).byteValue();
            }

            vectors.add(byteArray);
        }

        setTotalVectorsCountAdded(getTotalVectorsCountAdded() + vectors.size());

        accept(vectors);
    }
}
