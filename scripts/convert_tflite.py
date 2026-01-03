#!/usr/bin/env python3
"""
Convert ONNX wake word model to TFLite format for openWakeWord runtime.

This script requires specific package versions:
- tensorflow==2.11.0
- tensorflow_probability==0.19.0
- onnx
- onnx-tf

Usage:
    python convert_tflite.py --input model.onnx --output model.tflite
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX to TFLite")
    parser.add_argument("--input", type=str, required=True, help="Input ONNX model path")
    parser.add_argument("--output", type=str, help="Output TFLite model path (default: same name with .tflite)")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.tflite')

    print(f"Converting {input_path} to {output_path}...")

    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    # Load ONNX model
    print("Loading ONNX model...")
    onnx_model = onnx.load(str(input_path))

    print("Converting to TensorFlow...")
    tf_rep = prepare(onnx_model)

    # Export to SavedModel
    saved_model_path = input_path.parent / "saved_model"
    print(f"Exporting to SavedModel at {saved_model_path}...")
    tf_rep.export_graph(str(saved_model_path))

    # Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"TFLite model saved to {output_path}")
    print(f"Model size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
