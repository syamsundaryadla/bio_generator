from flask import Flask, request, jsonify, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__, template_folder="templates")

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

@app.route("/generate-bio", methods=["POST"])
def generate_bio():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Build the prompt
        prompt = (
            f"My name is {data['name']}. I am {data['age']} years old. "
            f"I am a {data['gender']} interested in {data['interests']}. "
            f"I work as a {data['profession']}."
        )

        # Tokenize input and generate response
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_return_sequences=1,
            temperature=0.8,  # Adjust creativity
            top_p=0.9,        # Use nucleus sampling
            repetition_penalty=2.0,  # Penalize repetition
            pad_token_id=tokenizer.eos_token_id,
        )
        
        # Decode and post-process the output
        bio = tokenizer.decode(outputs[0], skip_special_tokens=True)
        bio = bio.split(".")  # Split sentences to remove unnecessary additions
        bio = ". ".join(bio[:len(prompt.split(".")) + 1])  # Limit to a coherent response

        return jsonify({"bio": bio})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Ensure index.html is in the templates folder

if __name__ == "__main__":
    app.run(debug=True)
