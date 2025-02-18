/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;

public class JVectorCodec extends FilterCodec {

    public static final String CODEC_NAME = "JVectorCodec";
    public final boolean quantized;

    public JVectorCodec() {
        this(CODEC_NAME, new Lucene101Codec(), false);
    }

    public JVectorCodec(boolean quantized) {
        this(CODEC_NAME, new Lucene101Codec(), quantized);
    }

    public JVectorCodec(String codecName, Codec delegate, boolean quantized) {
        super(codecName, delegate);
        this.quantized = quantized;
    }

    @Override
    public KnnVectorsFormat knnVectorsFormat() {
        return new JVectorFormat(quantized);
    }

    @Override
    public CompoundFormat compoundFormat() {
        return new JVectorCompoundFormat(delegate.compoundFormat());
    }
}
