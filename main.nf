#!/usr/bin/env nextflow

params.run_embedding = false
params.output        = 'output_dir'

// Raw input files (used only if embedding = true)
params.raw_train     = 'data/train.csv'
params.raw_test      = 'data/test.csv'
params.raw_lnc       = 'data/lncrna.csv'
params.raw_prot      = 'data/protein.csv'

// Embedded input files (used if embedding = false)
params.train         = 'embeddings/train_emb_len100.csv'
params.test          = 'embeddings/test_emb_len100.csv'
params.predict_lnc   = 'embeddings/lncrna_emb_len100.csv'
params.predict_prot  = 'embeddings/protein_emb_len100.csv'

workflow {

    if (params.run_embedding) {
        process_embed("train", params.raw_train)
        process_embed("test", params.raw_test)
        process_embed("lnc", params.raw_lnc)
        process_embed("prot", params.raw_prot)
    }

    process_train(params.train, params.output)
    process_test(params.test, params.output)
    process_predict_lnc(params.predict_lnc, params.output)
    process_predict_prot(params.predict_prot, params.output)
}

process process_embed {
    container 'glncddr:latest'
    tag "$sample"

    input:
    val sample
    val file_path

    output:
    file "embeddings/${sample}_emb_len100.csv" into embedded_files

    script:
    """
    python /app/ml_pipeline/embedding/run_embedding.py \\
      --input $file_path \\
      --output embeddings/${sample}_emb_len100.csv \\
      --walks 5 --length 10 --dim 100
    """
}

process process_train {
    container 'glncddr:latest'

    input:
    val train_file
    val out_dir

    script:
    """
    python /app/main.py \\
      --train $train_file \\
      --test ${params.test} \\
      --predict_lnc ${params.predict_lnc} \\
      --predict_prot ${params.predict_prot} \\
      --output $out_dir
    """
}

process process_test {
    container 'glncddr:latest'

    input:
    val test_file
    val out_dir

    script:
    """
    python /app/main.py \\
      --train ${params.train} \\
      --test $test_file \\
      --predict_lnc ${params.predict_lnc} \\
      --predict_prot ${params.predict_prot} \\
      --output $out_dir
    """
}

process process_predict_lnc {
    container 'glncddr:latest'

    input:
    val lnc_file
    val out_dir

    script:
    """
    python /app/main.py \\
      --train ${params.train} \\
      --test ${params.test} \\
      --predict_lnc $lnc_file \\
      --predict_prot ${params.predict_prot} \\
      --output $out_dir
    """
}

process process_predict_prot {
    container 'glncddr:latest'

    input:
    val prot_file
    val out_dir

    script:
    """
    python /app/main.py \\
      --train ${params.train} \\
      --test ${params.test} \\
      --predict_lnc ${params.predict_lnc} \\
      --predict_prot $prot_file \\
      --output $out_dir
    """
}
