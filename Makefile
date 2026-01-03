.PHONY: build download train deploy clean help

WAKE_WORD ?= hey assistant
OUTPUT_DIR ?= /data/output
DATA_DIR ?= $(PWD)/data

help:
	@echo "openWakeWord Trainer"
	@echo ""
	@echo "Usage:"
	@echo "  make build              Build Docker image"
	@echo "  make download           Download training datasets"
	@echo "  make download-minimal   Download only essential features"
	@echo "  make train              Train wake word model"
	@echo "  make all                Full pipeline (download + train)"
	@echo "  make clean              Remove output files"
	@echo ""
	@echo "Variables:"
	@echo "  WAKE_WORD    Wake word phrase (default: 'hey assistant')"
	@echo "  OUTPUT_DIR   Output directory (default: /data/output)"
	@echo "  DATA_DIR     Host data directory (default: ./data)"
	@echo ""
	@echo "Examples:"
	@echo "  make train WAKE_WORD='ok computer'"
	@echo "  make all WAKE_WORD='hey jarvis'"

build:
	docker build -t openwakeword-trainer .

download:
	mkdir -p $(DATA_DIR)
	docker run --rm \
		-v $(DATA_DIR):/data \
		openwakeword-trainer \
		python /app/scripts/download_data.py \
			--output-dir /data \
			--datasets all

download-minimal:
	mkdir -p $(DATA_DIR)
	docker run --rm \
		-v $(DATA_DIR):/data \
		openwakeword-trainer \
		python /app/scripts/download_data.py \
			--output-dir /data \
			--datasets features rirs

train:
	docker run --gpus all --rm \
		-v $(DATA_DIR):/data \
		openwakeword-trainer \
		python /app/scripts/train.py \
			--wake-word "$(WAKE_WORD)" \
			--output-dir $(OUTPUT_DIR)

train-step-%:
	docker run --gpus all --rm \
		-v $(DATA_DIR):/data \
		openwakeword-trainer \
		python /app/scripts/train.py \
			--config $(OUTPUT_DIR)/$$(echo "$(WAKE_WORD)" | tr ' ' '_' | tr '[:upper:]' '[:lower:]').yml \
			--step $*

all: build download train
	@echo ""
	@echo "Training complete!"
	@echo "Model: $(DATA_DIR)/output/$$(echo '$(WAKE_WORD)' | tr ' ' '_' | tr '[:upper:]' '[:lower:]').tflite"

clean:
	rm -rf $(DATA_DIR)/output
