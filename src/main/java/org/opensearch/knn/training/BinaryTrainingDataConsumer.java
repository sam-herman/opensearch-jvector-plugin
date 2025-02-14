/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.training;

import lombok.extern.log4j.Log4j2;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.search.SearchHit;

import java.util.ArrayList;
import java.util.List;

/**
 * Transfers binary vectors from JVM to native memory.
 */
@Log4j2
public class BinaryTrainingDataConsumer extends TrainingDataConsumer {

    /**
     * Constructor
     *
     */
    public BinaryTrainingDataConsumer() {
        super();
    }

    @Override
    public void accept(List<?> byteVectors) {}

    @Override
    public void processTrainingVectors(SearchResponse searchResponse, int vectorsToAdd, String fieldName) {
        SearchHit[] hits = searchResponse.getHits().getHits();
        List<byte[]> vectors = new ArrayList<>();
        String[] fieldPath = fieldName.split("\\.");
        int nullVectorCount = 0;

        for (int vector = 0; vector < vectorsToAdd; vector++) {
            Object fieldValue = extractFieldValue(hits[vector], fieldPath);
            if (fieldValue == null) {
                nullVectorCount++;
                continue;
            }

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

        if (nullVectorCount > 0) {
            log.warn("Found {} documents with null byte vectors in field {}", nullVectorCount, fieldName);
        }

        setTotalVectorsCountAdded(getTotalVectorsCountAdded() + vectors.size());

        accept(vectors);
    }
}
