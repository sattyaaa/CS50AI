## Layer 3, Head 1
This attention head primarily focuses on the **next word** in the sequence. Instead of attending to itself, each token distributes most of its attention to the token that follows it. This pattern helps the model capture **local sequential dependencies**, which is important for predicting upcoming words and maintaining the flow of the sentence.

**Example Sentences:**
- The quick brown fox jumps over the lazy dog.
- Artificial intelligence is a fascinating field of study.

---

## Layer 12, Head 4
This head primarily focuses on the **period `.`** at the end of the sentence. Nearly every token attends to the final punctuation, which allows the model to **understand sentence boundaries** and gather the complete context of the sentence. This behavior typically occurs in the last layers, where the model consolidates token-level information into sentence-level understanding.

**Example Sentences:**
- After the rain stopped, we went for a walk in the park.
- The best way to predict the future is to `[MASK]` it.
