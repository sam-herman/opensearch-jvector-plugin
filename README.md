[![Build and Test k-NN](https://github.com/opensearch-project/k-NN/actions/workflows/CI.yml/badge.svg)](https://github.com/opensearch-project/k-NN/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/opensearch-project/k-NN/branch/main/graph/badge.svg?token=PYQO2GW39S)](https://codecov.io/gh/opensearch-project/k-NN)
[![Documentation](https://img.shields.io/badge/doc-reference-blue)](https://opensearch.org/docs/search-plugins/knn/index/)
[![Chat](https://img.shields.io/badge/chat-on%20forums-blue)](https://forum.opensearch.org/c/plugins/k-nn/48)
![PRs welcome!](https://img.shields.io/badge/PRs-welcome!-success)

# opensearch-jvector-plugin
- [Welcome!](#welcome)
- [Project Resources](#project-resources)
- [Credits and  Acknowledgments](#credits-and-acknowledgments)
- [Code of Conduct](#code-of-conduct)
- [License](#license)
- [Copyright](#copyright)

## Welcome!

**OpenSearch jVector Plugin** enables you to run the nearest neighbor search on billions of documents across thousands of dimensions with the same ease as running any regular OpenSearch query. You can use aggregations and filter clauses to further refine your similarity search operations. k-NN similarity search powers use cases such as product recommendations, fraud detection, image and video search, related document search, and more.

### Why use OpenSearch jVector Plugin?

#### High Level
- **Scalable**: Run similarity search on billions of documents across thousands of dimensions without exceeding memory by using DiskANN
- **Fast**: Blazing fast pure Java implementation with minimal overhead (see [benchmarks](https://github.com/jbellis/jvector/blob/main/README.md))
- **Lightweight**: Pure Java implementation. Self-contained, builds in seconds, no need to deal with native dependencies and complex flaky builds.

#### Unique Features
- **DiskANN**: JVector is capable to perform search without loading the entire index into RAM. This is a functionality that is not available today through Lucene and can be done through jVector without involving native dependencies (FAISS) and cumbersome JNI mechanism.
- _**Thread Safety**_ - JVector is a threadsafe index that supports concurrent modification and inserts with near perfect scalability as you add cores, Lucene is not threadsafe; OpenSearch kind of works around this with multiple segments but then has to compact them so insert performance still suffers (and I believe you can't read from a lucene segment during construction)
- _**quantized index construction**_ - JVector can perform index construction w/ quantized vectors, saving memory = larger segments = fewer segments = faster searches
- _**Quantized Disk ANN**_ - JVector supports DiskANN style quantization with rerank, it's quite easy (in principle) to demonstrate that this is a massive difference in performance for larger-than-memory indexes (in practice it takes days/weeks to insert enough vectors into Lucene to show this b/c of the single threaded problem, that's the only hard part)
- _**PQ and BQ support**_  - As part of (3) JVector supports PQ as well as the BQ that Lucene offers, it seems that this is fairly rare (pgvector doesn't do PQ either) because (1) the code required to get high performance ADC with SIMD is a bit involved and (2) it requires a separate codebook which Lucene isn't set up to easily accommodate.  PQ at 64x compression gives you higher relevance than BQ at 32x
- _**Fused ADC**_ - Features that nobody else has like Fused ADC and NVQ and Anisotropic PQ
- _**Compatibility**_ - JVector is compatible with Cassandra. Which allows to more easily transfer vector encoded data from Cassandra to OpenSearch and vice versa.

## Project Resources

* [Project Website](https://opensearch.org/)
* [Downloads](https://opensearch.org/downloads.html).
* [Documentation](https://opensearch.org/docs/search-plugins/knn/index/)
* Need help? Try the [Forum](https://forum.opensearch.org/c/plugins/k-nn/48)
* [Project Principles](https://opensearch.org/#principles)
* [Contributing to OpenSearch jVector Plugin](CONTRIBUTING.md)
* [Maintainer Responsibilities](MAINTAINERS.md)
* [Release Management](RELEASING.md)
* [Admin Responsibilities](ADMINS.md)
* [Security](SECURITY.md)

## Credits and Acknowledgments

This project uses two similarity search libraries to perform Approximate Nearest Neighbor Search: the Apache 2.0-licensed [Lucene](https://github.com/apache/lucene) and [jVector](https://github.com/jbellis/jvector).

## Code of Conduct

This project has adopted the [Placeholder Open Source Code of Conduct](CODE_OF_CONDUCT.md). For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq), or contact [placeholder](mailto:placeholder) with any additional questions or comments.

## License

This project is licensed under the [Apache v2.0 License](LICENSE.txt).

## Copyright

Copyright OpenSearch Contributors. See [NOTICE](NOTICE.txt) for details.