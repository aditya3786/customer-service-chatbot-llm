import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

import re

CS_ENTITIES = {
    "Algorithm/Model": [
        "transformer", "bert", "gpt", "gpt-2", "gpt-3", "gpt-4", "llm", "llms",
        "diffusion model", "diffusion", "stable diffusion", "gan", "generative adversarial",
        "vae", "variational autoencoder", "cnn", "convolutional neural network",
        "rnn", "recurrent neural network", "lstm", "gru", "attention mechanism",
        "self-attention", "multi-head attention", "encoder-decoder", "seq2seq",
        "vit", "vision transformer", "resnet", "alexnet", "vgg", "inception",
        "yolo", "faster r-cnn", "mask r-cnn", "detr", "t5", "roberta", "xlnet",
        "word2vec", "glove", "fasttext", "neural network", "deep learning",
        "reinforcement learning", "q-learning", "policy gradient", "actor-critic",
        "random forest", "gradient boosting", "xgboost", "svm", "support vector machine",
        "k-means", "dbscan", "pca", "autoencoder", "contrastive learning",
        "clip", "dall-e", "whisper", "llama", "mistral", "palm", "gemini",
    ],
    "Dataset": [
        "imagenet", "cifar-10", "cifar-100", "cifar", "mnist", "fashion-mnist",
        "coco", "ms coco", "voc", "pascal voc", "ade20k", "cityscapes",
        "squad", "squad 2.0", "glue", "superglue", "snli", "mnli",
        "wikipedia", "bookcorpus", "pile", "c4", "openwebtext",
        "wmt", "europarl", "commoncrawl", "common crawl",
        "librispeech", "voxceleb", "audioset",
        "kinetics", "ucf101", "hmdb51",
        "celeba", "lfw", "wider face",
        "modelnet", "shapenet",
        "open images", "laion",
    ],
    "Task": [
        "image classification", "object detection", "semantic segmentation",
        "instance segmentation", "image segmentation", "depth estimation",
        "machine translation", "text classification", "sentiment analysis",
        "named entity recognition", "question answering", "text generation",
        "text summarization", "natural language inference", "coreference resolution",
        "speech recognition", "speech synthesis", "text-to-speech",
        "image generation", "image synthesis", "super resolution",
        "optical flow", "pose estimation", "action recognition",
        "knowledge graph", "link prediction", "node classification",
        "recommendation", "anomaly detection", "few-shot learning",
        "zero-shot learning", "transfer learning", "meta-learning",
        "multi-task learning", "continual learning", "federated learning",
        "visual question answering", "image captioning", "visual grounding",
    ],
    "Framework": [
        "pytorch", "tensorflow", "keras", "jax", "flax",
        "hugging face", "huggingface", "transformers library",
        "scikit-learn", "sklearn", "xgboost", "lightgbm",
        "langchain", "llama-index", "openai api",
        "cuda", "cudnn", "triton",
        "onnx", "tensorrt", "openvino",
        "ray", "deepspeed", "fsdp", "megatron",
        "wandb", "mlflow", "dvc",
    ],
}

CATEGORY_COLORS = {
    "Algorithm/Model": "#cce5ff",
    "Dataset": "#d4edda",
    "Task": "#fff3cd",
    "Framework": "#f8d7da",
}

_PATTERNS: dict[str, list[tuple[str, re.Pattern]]] = {}
for _cat, _terms in CS_ENTITIES.items():
    _PATTERNS[_cat] = [
        (term, re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE))
        for term in sorted(_terms, key=len, reverse=True)
    ]


def extract_cs_entities(text: str) -> list[dict]:
    entities = []
    covered: list[tuple[int, int]] = []

    for category, patterns in _PATTERNS.items():
        for term, pattern in patterns:
            for m in pattern.finditer(text):
                start, end = m.start(), m.end()
                if any(s <= start < e or s < end <= e for s, e in covered):
                    continue
                entities.append({"term": m.group(), "category": category, "start": start, "end": end})
                covered.append((start, end))

    return sorted(entities, key=lambda x: x["start"])


def highlight_cs_entities(text: str, entities: list[dict]) -> str:
    if not entities:
        return text
    result = []
    prev = 0
    for ent in sorted(entities, key=lambda x: x["start"]):
        result.append(text[prev:ent["start"]])
        color = CATEGORY_COLORS.get(ent["category"], "#e0e0e0")
        result.append(
            f'<mark style="background:{color};padding:1px 4px;border-radius:3px">'
            f'{ent["term"]}</mark>'
        )
        prev = ent["end"]
    result.append(text[prev:])
    return "".join(result)
