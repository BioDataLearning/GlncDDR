#!/usr/bin/env nextflow

params.output        = 'output_dir'
params.train         = 'embeddings/train_emb_len100.csv'
params.test          = 'embeddings/test_emb_len100.csv'
params.predict_lnc   = 'embeddings/lncrna_emb_len100.csv'
params.predict_prot  = 'embeddings/protein_emb_len100.csv'

workflow {
    process_train(
        file(params.train),
        file(params.test),
        file(params.predict_lnc),
        file(params.predict_prot),
        params.output
    )
    process_test(
        file(params.train),
        file(params.test),
        file(params.predict_lnc),
        file(params.predict_prot),
        params.output
    )
    process_predict_lnc(
        file(params.train),
        file(params.test),
        file(params.predict_lnc),
        file(params.predict_prot),
        params.output
    )
    process_predict_prot(
        file(params.train),
        file(params.test),
        file(params.predict_lnc),
        file(params.predict_prot),
        params.output
    )
}

process process_train {
    container 'glncddr:latest'

    input:
    path train_file
    path test_file
    path lnc_file
    path prot_file
    val out_dir

    script:
    """
    python /app/main.py \\
      --train $train_file \\
      --test $test_file \\
      --predict_lnc $lnc_file \\
      --predict_prot $prot_file \\
      --output $out_dir
    """
}

process process_test {
    container 'glncddr:latest'

    input:
    path train_file
    path test_file
    path lnc_file
    path prot_file
    val out_dir

    script:
    """
    python /app/main.py \\
      --train $train_file \\
      --test $test_file \\
      --predict_lnc $lnc_file \\
      --predict_prot $prot_file \\
      --output $out_dir
    """
}

process process_predict_lnc {
    container 'glncddr:latest'

    input:
    path train_file
    path test_file
    path lnc_file
    path prot_file
    val out_dir

    script:
    """
    python /app/main.py \\
      --train $train_file \\
      --test $test_file \\
      --predict_lnc $lnc_file \\
      --predict_prot $prot_file \\
      --output $out_dir
    """
}

process process_predict_prot {
    container 'glncddr:latest'

    input:
    path train_file
    path test_file
    path lnc_file
    path prot_file
    val out_dir

    script:
    """
    python /app/main.py \\
      --train $train_file \\
      --test $test_file \\
      --predict_lnc $lnc_file \\
      --predict_prot $prot_file \\
      --output $out_dir
    """
}
