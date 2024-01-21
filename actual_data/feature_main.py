import argparse
import pickle
import gzip
from feature import HotpotDataReader,HotpotFeatureConverter

class Config:
    max_document_size = 384
    
def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--entity_path", required=True, type=str)
    parser.add_argument("--para_path", required=True, type=str)
    parser.add_argument("--example_output", required=True, type=str)
    parser.add_argument("--feature_output", required=True, type=str)

    # Other parameters
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--full_data", type=str, required=True)
    parser.add_argument("--word2idx_file", default="/home/multihop_question_generation/hotpot/word2idx.pkl")
    parser.add_argument("--vocab_size", type=int, default=45000)

    args = parser.parse_args()

    with open(args.word2idx_file, "rb") as f:
        word2idx = pickle.load(f)

    data_reader = HotpotDataReader(para_file=args.para_path, full_file=args.full_data, entity_file=args.entity_path)
    examples = data_reader.read_hotpot_examples_QG()
    
    with gzip.open(args.example_output, 'wb') as fout:
        pickle.dump(examples, fout)

    convert_features = HotpotFeatureConverter(Config())
    features = convert_features.convert_examples_to_features_QG(examples, word2idx, max_seq_length=args.max_seq_length, max_query_length=50)

    with gzip.open(args.feature_output, 'wb') as fout:
        pickle.dump(features, fout)

if __name__ == "__main__":
    main()