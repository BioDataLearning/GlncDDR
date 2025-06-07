import argparse
import os
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import MinMaxScaler
from ml_pipeline.scripts.training import train_models
from ml_pipeline.scripts.testing import test_models
from ml_pipeline.scripts.predict_lnc import predict_lncRNA
from ml_pipeline.scripts.predict_prot import predict_protein
from ml_pipeline.scripts.utils import get_training_data, set_global_seed
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def main():
    print("🚀 main() started")
    parser = argparse.ArgumentParser(description="Run DDR ML Pipeline")
    parser.add_argument('--train', required=True, help="Path to training CSV file")
    parser.add_argument('--test', required=True, help="Path to test CSV file")
    parser.add_argument('--predict_lnc', required=True, help="Path to lncRNA prediction CSV file")
    parser.add_argument('--predict_prot', required=True, help="Path to protein-coding prediction CSV file")
    parser.add_argument('--output', required=True, help="Output directory path")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print("📁 Output directory ensured:", args.output)

    set_global_seed(42)
    print("🌱 Global seed set")

    x_exp, _ = get_training_data(args.train)
    scaler = MinMaxScaler()
    scaler.fit(x_exp)
    print("📐 Scaler fitted")

    models = train_models(scaler, args.train, args.output)
    print("🤖 Models trained")

    test_models(models, scaler, args.test, args.output)
    print("🧪 Models tested")

    predict_lncRNA(models, scaler, args.predict_lnc, args.output)
    predict_protein(models, scaler, args.predict_prot, args.output)
    print("✅ All done!")

if __name__ == '__main__':
    main()
