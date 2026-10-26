# Summary Report — Deep Learning Practical Report 6

## Title
**Opening the Transformer: Attention, Encoder, Decoder and Inference with Pretrained Models**

## Objective
To understand how Transformer architectures represent token relationships and to inspect pretrained encoder-only, decoder-only and encoder-decoder models. The practical uses Hugging Face models and NumPy implementations; no training or fine-tuning is required.

## Models and tools
- `distilbert-base-uncased-finetuned-sst-2-english` — sentiment analysis.
- `distilbert-base-uncased` — fill-mask and encoder attention inspection.
- `gpt2` — causal self-attention and autoregressive generation.
- `t5-small` — translation, summarization and cross-attention.
- Python libraries: Transformers, PyTorch, Datasets, NumPy, Pandas, Matplotlib, Seaborn and scikit-learn.

## Work completed in the notebook design
1. **Pipelines and tokenization:** Run sentiment inference, inspect token IDs and attention masks, and print model configuration.
2. **Self-attention:** Implement Query, Key and Value projections and scaled dot-product attention; compare scaling and Luong dot attention.
3. **Multi-head attention and positions:** Visualize attention heads, build sinusoidal positional encodings, inspect learned position embeddings, and compare LayerNorm with BatchNorm.
4. **Encoder and masking:** Inspect encoder components, run fill-mask predictions, and compare bidirectional attention with causal masking.
5. **Decoder and inference:** Use T5 for translation/summarization and GPT-2 for greedy decoding, decoding strategies, and KV-cache timing.
6. **Evaluation and mini application:** Evaluate sentiment on a 200-review IMDB sample at multiple maximum sequence lengths, measure inference timing, inspect factual-generation outputs, and build a sentiment-summary-translation toolkit.

## Key concepts
- Attention weights indicate how strongly a token representation uses information from other tokens.
- Scaling dot products by the square root of key dimension helps keep softmax scores in a useful range.
- Positional information is needed because self-attention alone does not distinguish different token orders.
- Encoder-only models are suitable for understanding/classification tasks; decoder-only models generate text autoregressively; encoder-decoder models support sequence-to-sequence tasks.
- Causal masks prevent a decoder token from accessing future positions.
- Longer inputs can increase compute and memory costs, while truncation can remove useful review context.
- Generative models may produce incorrect factual statements, so outputs should be checked.

## Graphs included
- Scaled dot-product attention matrix.
- Causal self-attention mask.
- Sinusoidal positional encoding heatmap.
- Schematic cross-attention alignment.
- Transformer family architecture overview.

## Results and limitations
The accompanying graphs are labelled illustrative or schematic where they are not produced by a pretrained-model run. Actual IMDB accuracy, precision, recall, F1, confusion matrix, timing measurements, and real model attention plots must be captured by running the notebook. No numerical experimental results are claimed in this report before execution.

## Conclusion
This practical connects the mathematical attention equations with real pretrained Transformer models. It demonstrates the roles of tokenization, attention, positional information, normalization, masking, cross-attention, and autoregressive inference. The mini text toolkit shows how pretrained models can be combined for sentiment analysis, summarization, and translation without training a new model.
