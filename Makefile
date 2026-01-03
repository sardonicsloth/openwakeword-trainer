.PHONY: build generate train convert all clean

WAKE_WORD ?= assistant
SAMPLE_COUNT ?= 5000
EPOCHS ?= 100

build:
	docker build -t openwakeword-trainer .

generate:
	docker run --gpus all --rm \
		-v $(PWD)/data:/data \
		openwakeword-trainer \
		python /data/scripts/generate_samples.py \
			--wake-word "$(WAKE_WORD)" \
			--count $(SAMPLE_COUNT) \
			--output-dir /data/output/samples

train:
	docker run --gpus all --rm \
		-v $(PWD)/data:/data \
		-v $(PWD)/scripts:/data/scripts \
		openwakeword-trainer \
		python /data/scripts/train_model.py \
			--wake-word "$(WAKE_WORD)" \
			--samples-dir /data/output/samples \
			--output-dir /data/output/model \
			--epochs $(EPOCHS)

convert:
	docker run --rm \
		-v $(PWD)/data/output/model:/model \
		python:3.10 bash -c "\
			pip install numpy==1.23.5 onnx onnx-tf tensorflow==2.11.0 tensorflow_probability==0.19.0 && \
			python -c \"\
import onnx; \
from onnx_tf.backend import prepare; \
import tensorflow as tf; \
onnx_model = onnx.load('/model/$(WAKE_WORD).onnx'); \
tf_rep = prepare(onnx_model); \
tf_rep.export_graph('/model/saved_model'); \
converter = tf.lite.TFLiteConverter.from_saved_model('/model/saved_model'); \
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]; \
tflite_model = converter.convert(); \
open('/model/$(WAKE_WORD).tflite', 'wb').write(tflite_model); \
print('Saved $(WAKE_WORD).tflite')\
\""

all: build generate train convert
	@echo "Wake word model ready: data/output/model/$(WAKE_WORD).tflite"

clean:
	rm -rf data/output
