nextflow.enable.dsl=2

params.input         = params.input
params.output_dir    = params.output_dir ?: "output_dir"
params.vector_size   = params.vector_size ?: 100
params.walks         = params.walks ?: 10
params.length        = params.length ?: 10
params.batch_id      = params.batch_id ?: "1"
params.reuse_walks   = params.reuse_walks ?: false

process node2vec_embedding {
    container 'python:3.9'
    time'24h'
    containerOptions '-w $(pwd)'
    input:
        tuple path(input_csv), path(script)
    output:
        path "train_embedding_${params.vector_size}.csv"
    script:
    """
    pip install --no-cache-dir pandas numpy networkx tqdm scikit-learn PyWGCNA gensim
    echo "SCRIPT IS: ./${script}"
    python ./${script} \
        --input ./${input_csv} \
        --output-dir ${params.output_dir} \
        --vector-size ${params.vector_size} \
        --walks ${params.walks} \
        --length ${params.length} \
        --batch-id ${params.batch_id}
    """
}

workflow {
    in_csv_ch   = Channel.fromPath(params.input, checkIfExists: true)
    script_ch   = Channel.fromPath('ml_pipeline/embedding/run_embedding.py', checkIfExists: true)
    pair_ch     = in_csv_ch.combine(script_ch)
    node2vec_embedding(pair_ch)
}