# -*- coding: UTF-8

from __future__ import print_function  # enable python3 printing
from time import gmtime, strftime

from Bio import SeqIO
from tensorflow.keras.models import load_model
from keras_preprocessing import sequence
import sys
import os
from attention import Attention

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # turns off tensorflow SSE compile warnings


# 模型四调用
def for_four(fa_path):
    print("STARTING JOB: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    # fname = sys.argv[1]  # Input FASTA file of peptides
    fname = fa_path  # Input FASTA file of peptides
    # model = "./TrainedModels/weights.h5"  # Saved Keras model in HDF5 format'
    model = "./model/weights.h5"  # Saved Keras model in HDF5 format'
    basefile = os.path.basename(fname)
    print(basefile)
    basename = os.path.splitext(basefile)[0]
    thrsh = 0.5  # greater than this value = AM
    # LOAD SEQUENCES
    amino_acids = "XACDEFGHIKLMNPQRSTVWY"
    aa2int = dict((c, i) for i, c in enumerate(amino_acids))
    X_test = []
    y_test = []
    warn = []
    ids = []
    seqs = []
    max_length = 200

    print("Encoding sequences...")
    for s in SeqIO.parse(fname, "fasta"):
        if (len(str(s.seq)) < 6 or len(str(s.seq)) > 200):
            warn.append('*')
        else:
            warn.append('')

        ids.append(str(s.id))
        seqs.append(str(s.seq))
        X_test.append([aa2int[aa] for aa in str(s.seq).upper()])

    X_test = sequence.pad_sequences(X_test, maxlen=max_length)

    # LOAD MODEL
    print("Loading model and weights for file: " + model)
    # loaded_model = load_model(model)
    loaded_model = load_model(model, custom_objects={'Attention': Attention})
    os.makedirs('hem', exist_ok=True)
    basename = 'hem/' + basename
    # PREDICT AND SAVE
    print("Making predictions and saving results...")
    fcand = open(basename + '_hemolysisCandidates.fa', 'w')
    fpsum = open(basename + '_Prediction_Summary.csv', 'w')
    fpsum.write("SeqID,Prediction_Class,Prediction_Probability,Sequence\n")

    preds = loaded_model.predict(X_test)
    data = {
        'optimal': [],
        'all': []
    }
    for i, pred in enumerate(preds):
        data['all'].append({
            'Sequence': seqs[i],
            'Hemolytic_model_score': '{:.4f}'.format(pred[0]),
        })
        if (pred[0] >= thrsh):
            data['optimal'].append({
                'Sequence': seqs[i],
                'Hemolytic_model_score': '{:.4f}'.format(pred[0]),
            })
            fpsum.write("{},Non-Hemolytic{},{},{}\n".format(ids[i], warn[i], round(pred[0], 4), seqs[i]))
            fcand.write(">{}\n{}\n".format(ids[i], seqs[i]))
        else:
            fpsum.write("{},Hemolytic{},{},{}\n".format(ids[i], warn[i], round(pred[0], 4), seqs[i]))

    fcand.close()
    fpsum.close()

    print("Saved files: " + basename + "_Prediction_Summary.csv, " + basename + "_AMPCandidates.fa")
    print("JOB FINISHED: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    return data
