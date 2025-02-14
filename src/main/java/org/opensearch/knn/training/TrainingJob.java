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

import lombok.Getter;
import org.apache.commons.lang.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.common.UUIDs;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;

import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.Objects;

/**
 * Encapsulates all information required to generate and train a model.
 */
public class TrainingJob implements Runnable {

    public static Logger logger = LogManager.getLogger(TrainingJob.class);

    private final KNNMethodContext knnMethodContext;
    private final KNNMethodConfigContext knnMethodConfigContext;
    @Getter
    private final Model model;

    @Getter
    private final String modelId;

    /**
     * Constructor.
     *
     * @param modelId String to identify model. If null, one will be generated.
     * @param knnMethodContext Method definition used to construct model.
     * @param description user provided description of the model.
     */
    public TrainingJob(
        String modelId,
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        String description,
        String nodeAssignment,
        Mode mode,
        CompressionLevel compressionLevel
    ) {
        // Generate random base64 string if one is not provided
        this.modelId = StringUtils.isNotBlank(modelId) ? modelId : UUIDs.randomBase64UUID();
        this.knnMethodContext = Objects.requireNonNull(knnMethodContext, "MethodContext cannot be null.");
        this.knnMethodConfigContext = knnMethodConfigContext;
        this.model = new Model(
            new ModelMetadata(
                knnMethodContext.getKnnEngine(),
                knnMethodContext.getSpaceType(),
                knnMethodConfigContext.getDimension(),
                ModelState.TRAINING,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                description,
                "",
                nodeAssignment,
                knnMethodContext.getMethodComponentContext(),
                knnMethodConfigContext.getVectorDataType(),
                mode,
                compressionLevel,
                knnMethodConfigContext.getVersionCreated()
            ),
            null,
            this.modelId
        );
    }

    @Override
    public void run() {
        throw new UnsupportedOperationException("Unsupported operation");
    }
}
