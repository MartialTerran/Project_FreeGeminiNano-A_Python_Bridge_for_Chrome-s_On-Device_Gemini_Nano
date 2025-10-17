
---

# Project Free_Gemini_Nano (Project Heimdall): A Python Bridge for Gemini Nano

**Our mission: To unlock Chrome's on-device Gemini Nano model, making its weights and architecture directly accessible from Python for advanced research and application development.**

This project is code-named Project Heimdall, after the Bifröst bridge guardian, because our goal is to create a bridge between two distinct realms: the sandboxed, high-performance environment of the Chrome browser and the flexible, powerful machine learning ecosystem of Python.

The goal is to break the model out of its sandbox for greater flexibility, research, and extension. This includes implementing cutting-edge techniques like **Artificial Hippocampus Networks (AHN)** to efficiently manage and expand the model's effective context length, overcoming the current token limits.

- **Learn more about AHN:** [Artificial Hippocampus Networks for Efficient Long-Context Modeling (arXiv)](https://arxiv.org/pdf/2510.07318)
- **Educational Implementation:** [Pure PyTorch Qwen2 Model with AHN by Martial Terran](https://huggingface.co/MartialTerran/Toy_Qwen2-AHN_ByteDance-Seed_AHN/blob/main/README.md)

## The Vision: On-Device AI, On Your Terms

Google has integrated a powerful, efficient version of its Gemini model (**Gemini Nano**) that runs entirely on-device within the Chrome browser. This offers unparalleled privacy and zero API costs but is currently restricted to a JavaScript API (`window.ai`).

Our vision is to liberate this model, enabling two primary pathways for Python developers:

1.  **Direct Python Inference (The Ultimate Goal):** Load the Gemini Nano model files (`weights.bin`) directly into a Python script and run inferences using standard ML frameworks. This would open the door to advanced research, fine-tuning, and integration with other Python-based tools.
2.  **Seamless WebGPU Bridging (The Pragmatic Bridge):** Create a high-performance bridge where Python scripts can send prompts to Chrome, have the browser execute the model using its optimized WebGPU engine, and receive the results back in Python.

## The Core Challenges

This is a reverse-engineering effort. The Gemini Nano model files, found in Chrome's protected user data directory, are not in a standard, portable format. To use them, we must overcome several significant challenges:

-   **Proprietary Model Format:** The `weights.bin` file is a compiled and quantized artifact, designed exclusively for Chrome's internal ONNX inference engine. It cannot be loaded by standard libraries like TensorFlow or PyTorch.
-   **Missing Preprocessing Logic:** The model expects input data (text, images, audio) to be converted into a precise numerical tensor format. This logic for resizing, normalizing, and ordering color channels is hidden within Chrome's C++ source code.
-   **Missing Postprocessing Logic:** The model's output is a tensor of raw probabilities (logits), not text. Reconstructing the decoding pipeline—which turns these numbers into human-readable words using auxiliary files (`.binarypb`, `.fst`, `.syms`)—is essential.

## How You Can Contribute

This is a complex project, and we need your help! We're looking for collaborators with expertise in:

-   **ML Reverse Engineering:** Individuals experienced in analyzing and converting proprietary model formats.
-   **Chromium Source Code Experts:** Developers comfortable navigating the [Chromium C++ codebase](https://source.chromium.org/chromium/chromium/src/+/main:components/optimization_guide/core/model_execution/on_device_model_service_controller.cc) to find the pre/post-processing logic.
-   **Python ML Frameworks:** Developers skilled in TensorFlow, PyTorch, and ONNX Runtime to help build the Python-side interface.
-   **WebGPU & Browser Automation:** For our bridging goal, we need experts in high-performance communication between Python and JS (e.g., via Playwright and WebSockets).

## Project Roadmap & First Steps

### Phase 1: The WebGPU Bridge (Pragmatic Track)
-   [ ] Develop a stable WebSocket-based server in Python.
-   [ ] Create a minimal HTML/JS client that can receive prompts, execute them via `window.ai`, and stream results back.
-   [ ] Bundle this into a user-friendly Python library.

### Phase 2: The Direct Inference Engine (Ambitious Track)
-   [ ] **Locate and Analyze Preprocessing Code:** Dive into the Chromium source to find the functions that prepare data for the model.
-   [ ] **Reverse-Engineer the `weights.bin` Format:** Analyze the binary format of the model weights and graph.
-   [ ] **Rebuild the Decoder:** Re-implement the output postprocessing logic in Python to turn logits into text.
-   [ ] **Build a Python Loader:** Create a Python class capable of loading the weights and running an inference step.

## Getting Started: A Script to Find the Model

This script will help you locate and copy the Gemini Nano model files from your local Chrome installation. This is the first step for any analysis.

**`find_nano.py`**
```python
import os
import shutil
import platform
import json

def find_latest_model_version(model_base_path: str) -> str | None:
    """Finds the directory with the highest version number."""
    try:
        subfolders = [f for f in os.listdir(model_base_path) if f.isdigit()]
        return sorted(subfolders, key=int, reverse=True)[0] if subfolders else None
    except (FileNotFoundError, IndexError):
        return None

def extract_model_info(destination_folder: str = "gemini_nano_files"):
    """
    Finds and copies the latest Gemini Nano model from Chrome's user data
    directory and prints its metadata.
    """
    print("--- Project Heimdall: Gemini Nano Extractor ---")
    if platform.system() != "Windows":
        print("Error: This script is currently configured for Windows.")
        return

    local_app_data = os.getenv('LOCALAPPDATA')
    model_source_base = os.path.join(local_app_data, 'Google', 'Chrome', 'User Data', 'OnDeviceModel')

    version = find_latest_model_version(model_source_base)
    if not version:
        print("Status: Gemini Nano model not found. Ensure the feature is enabled in chrome://flags.")
        return

    source_path = os.path.join(model_source_base, version)
    print(f"Found model version {version} at: {source_path}")

    try:
        os.makedirs(destination_folder, exist_ok=True)
        shutil.copytree(source_path, destination_folder, dirs_exist_ok=True)
        print(f"Successfully copied model files to '{destination_folder}' directory.")
        
        metadata_path = os.path.join(destination_folder, 'model.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            print("\n--- Model Metadata ---")
            print(json.dumps(metadata, indent=2))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    extract_model_info()
```

## Conceptual Code for the WebGPU Bridge

This illustrates the basic logic for the Python-to-JavaScript communication.

**`bridge_server.py` (Conceptual Python Server):**
```python
# Requires 'websockets': pip install websockets
import asyncio
import websockets

async def handler(websocket, path):
    print("Browser connected.")
    try:
        # 1. Wait for a prompt from the Python user (this part would be expanded)
        prompt_to_run = "Describe this image in detail." # Example prompt
        
        # 2. Send prompt to the browser to be executed
        await websocket.send(prompt_to_run)
        
        # 3. Wait for the result back from the browser
        ai_result = await websocket.recv()
        print(f"Received result from Gemini Nano: {ai_result}")

    except websockets.ConnectionClosed:
        print("Browser connection closed.")

async def main():
    print("Starting WebSocket server on ws://localhost:8765")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
```

**`bridge_client.html` (Conceptual JavaScript Client):**
```html
<script>
    const socket = new WebSocket('ws://localhost:8765');

    socket.onopen = () => {
        console.log("Connected to Python bridge server.");
    };

    // 1. Listen for a prompt from the Python server
    socket.onmessage = async (event) => {
        const prompt = event.data;
        console.log(`Received prompt from Python: ${prompt}`);

        // 2. Run the prompt using the Built-in AI API
        const session = await window.ai.createTextSession();
        const result = await session.prompt(prompt);
        session.destroy();

        // 3. Send the result back to Python
        socket.send(result);
    };
</script>
```

---
**Disclaimer:** This is an independent, experimental project for educational and research purposes. The internal APIs and model formats of Chrome are undocumented, subject to change without notice, and are not intended for external use. This project is not affiliated with or endorsed by Google.
```
