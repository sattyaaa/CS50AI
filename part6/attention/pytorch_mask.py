import sys
import os
import torch

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, BertForMaskedLM
from transformers.tokenization_utils_base import BatchEncoding

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
# Make sure this font path is correct for your environment
try:
    FONT = ImageFont.truetype("/content/attention/assets/fonts/OpenSans-Regular.ttf", 28)
except IOError:
    print("Font file not found. Using default font.")
    FONT = ImageFont.load_default()
    
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input for PyTorch
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="pt") # Changed "tf" to "pt"
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use PyTorch model to process input
    model = BertForMaskedLM.from_pretrained(MODEL) # Changed from TFBertForMaskedLM
    
    # Put the model in evaluation mode
    model.eval()

    # Disable gradient calculations for inference
    with torch.no_grad():
        result = model(**inputs, output_attentions=True)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = torch.topk(mask_token_logits, K).indices # Changed from tf.math.top_k
    for token in top_tokens:
        # .item() gets the integer value from a 0-dim tensor
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token.item()])))

    # Visualize attentions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    visualize_attentions(tokens, result.attentions)


def get_mask_token_index(mask_token_id, inputs: BatchEncoding):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.
    """
    # This function works for PyTorch tensors as well without modification
    for i, token_id in enumerate(inputs.input_ids[0]):
        if token_id == mask_token_id:
            return i
    return None


def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """
    # This function is framework-agnostic
    color = int(attention_score * 255)
    return color, color, color


def visualize_attentions(tokens, attentions):
    """
    Produce a graphical representation of self-attention scores.

    For each attention layer, one diagram should be generated for each
    attention head in the layer. Each diagram should include the list of
    `tokens` in the sentence. The filename for each diagram should
    include both the layer number (starting count from 1) and head number
    (starting count from 1).
    """
    for i, layer in enumerate(attentions):
        # layer is a tensor of shape (batch_size, num_heads, seq_length, seq_length)
        num_heads = layer.shape[1]
        for k in range(num_heads):
            layer_number = i + 1
            head_number = k + 1
            
            # Detach tensor from graph and convert to numpy for PIL
            attention_weights = layer[0, k].detach().numpy()

            generate_diagram(
                layer_number,
                head_number,
                tokens,
                attention_weights
            )


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each of the
    `tokens`, and cells are shaded based on `attention_weights`, with lighter
    cells corresponding to higher attention scores.

    The diagram is saved with a filename that includes both the `layer_number`
    and `head_number`.
    """
    # Create new image
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):
        # Draw token columns
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Draw token rows
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw each word
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    # 1. Define the folder you want to save into
    output_folder = "attention_outputs"

    # 2. Create the folder if it doesn't already exist
    os.makedirs(output_folder, exist_ok=True)

    # 3. Create the full path by joining the folder and filename
    file_name = f"Attention_Layer{layer_number}_Head{head_number}.png"
    full_path = os.path.join(output_folder, file_name)

    # 4. Save the image to the full path
    img.save(full_path)
    print(f"Saved diagram to {full_path}")


if __name__ == "__main__":
    main()