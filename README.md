# Project_FreeGeminiNano-A_Python_Bridge_for_Chrome-s_On-Device_Gemini_Nano
An ambitious open-source initiative to unlock the full potential of Google's Gemini Nano model by making it directly accessible to Python developers. We aim to bridge the gap between the model's secure, sandboxed environment within Chrome and the powerful, flexible world of Python and its machine learning ecosystem.

The goal is to break the model out of its sandbox for greater flexibility and research, and extension such as implementing cutting-edge techniques like **Artificial Hippocampus Networks** (as described in [arXiv:2510.07318](https://arxiv.org/pdf/2510.07318)) to efficiently manage and expand the model's effective context length, overcoming the current token limits.

## The Vision: On-Device AI, On Your Terms

Google has deployed a powerful, efficient version of its Gemini model (Nano) that runs entirely on-device within the Chrome browser. This offers unparalleled privacy and zero API costs. However, its use is currently restricted to a JavaScript API (`window.ai`).

Our vision is to liberate this model, enabling two primary pathways for Python developers:

1.  **Direct Inference:** Load the Gemini Nano model files (`weights.bin`) directly into a Python script and run inferences using standard ML frameworks. This would open the door to advanced research, fine-tuning, and integration with other Python-based tools.
2.  **Seamless Bridging:** Create a high-performance bridge where Python scripts can send prompts to Chrome, have the browser execute the model using its optimized WebGPU engine, and receive the results back in Python.

## The Core Challenges

This is a reverse-engineering effort. The Gemini Nano model files, found in Chrome's user data directory, are not in a standard, portable format.

*   **Proprietary Model Format:** The `weights.bin` file is a compiled and quantized artifact, designed exclusively for Chrome's internal ONNX inference engine. It cannot be loaded by standard libraries like TensorFlow or PyTorch.
*   **Missing Preprocessing Logic:** The model expects input data (text, images, audio) to be converted into a precise numerical tensor format. This logic for resizing, normalizing, and ordering color channels is hidden within Chrome's C++ source code.
*   **Missing Postprocessing Logic:** The model's output is a tensor of raw probabilities (logits), not text. Reconstructing the decoding pipeline, which turns these numbers into human-readable words, is essential.

## Our Two-Pronged Approach

We are tackling this challenge on two fronts.

### Track 1 (Ambitious): Direct Python Inference

This is the ultimate goal. We aim to create a Python library that can load and run the `weights.bin` file directly. Success here would enable groundbreaking applications.

**Research Goal: Extending Context Length**
A primary motivation for direct access is to experiment with extending the model's capabilities. For instance, we could explore implementing cutting-edge techniques like **Artificial Hippocampus Networks** (as described in [arXiv:2510.07318](https://arxiv.org/pdf/2510.07318)) to efficiently manage and expand the model's effective context length, overcoming the current token limits.

### Track 2 (Pragmatic): Python-to-WebGPU Bridge

This is the more immediately achievable goal. By leveraging browser automation, we can build a robust bridge that allows Python to use Chrome as a local "inference server."

**How it Works:**
1.  A Python script starts a local WebSocket or HTTP server.
2.  The script uses a tool like Playwright to launch a dedicated Chrome window with a specialized HTML page.
3.  The Python script sends a prompt to the JavaScript running on that page.
4.  The JavaScript uses the `window.ai` API to execute the prompt against Gemini Nano, leveraging the GPU via **WebGPU** for maximum performance.
5.  The JavaScript sends the model's response back to the Python server.

This approach lets us use Google's highly optimized engine while still controlling the logic from Python.

## How You Can Help

We are looking for collaborators with expertise in:

*   **Machine Learning Reverse Engineering:** If you have experience analyzing proprietary model formats or weights, your skills are critical for Track 1.
*   **C++ & Chromium Source Code:** The secrets to the pre/post-processing logic are in the [Chromium source](https://source.chromium.org/chromium/chromium/src/+/main:components/optimization_guide/core/model_execution/on_device_model_service_controller.cc). We need developers who can navigate this codebase and translate the logic to Python.
*   **WebGPU & JavaScript Performance:** For Track 2, we need experts in high-performance browser automation and efficient communication between Python and JS (e.g., via WebSockets).
*   **Python ML Frameworks:** Developers skilled in TensorFlow, PyTorch, and ONNX Runtime to help build the Python-side interface.

## Getting Started & Code Snippets

### 1. Python Script to Find and Copy the Model

First, let's verify the model exists and make a local copy to work with. Save this as `nano_model.py`.

```python
import os
import shutil
import platform

def find_latest_model_version(model_base_path: str) -> str | None:
    try:
        subfolders = [f for f in os.listdir(model_base_path) if f.isdigit()]
        return sorted(subfolders, key=int, reverse=True) if subfolders else None
    except FileNotFoundError:
        return None

def copy_nano_files(destination_folder: str):
    """Finds and copies Gemini Nano model files to a specified folder."""
    print(f"Attempting to copy model to '{destination_folder}'...")
    local_app_data = os.getenv('LOCALAPPDATA')
    model_source_base = os.path.join(local_app_data, 'Google', 'Chrome', 'User Data', 'OnDeviceModel')

    version = find_latest_model_version(model_source_base)
    if not version:
        print("Gemini Nano model not found in Chrome directory.")
        return

    source_path = os.path.join(model_source_base, version)
    try:
        shutil.copytree(source_path, destination_folder, dirs_exist_ok=True)
        print(f"Successfully copied model version {version} to destination.")
    except Exception as e:
        print(f"An error occurred during copy: {e}")

if __name__ == "__main__":
    copy_nano_files("gemini_nano_files")
2. Conceptual Code for the WebGPU Bridge

This illustrates the logic for the Python-to-JavaScript communication.

Python Server (bridge_server.py - Conceptual):

code
Python
download
content_copy
expand_less
# This would be a WebSocket server using a library like 'websockets'
# pip install websockets

import asyncio
import websockets

async def handler(websocket, path):
    print("Browser connected.")
    try:
        # 1. Wait for a prompt from the Python user
        prompt_to_run = "Describe this image in detail." # This would be dynamic
        
        # 2. Send prompt to the browser to be executed
        await websocket.send(prompt_to_run)
        
        # 3. Wait for the result back from the browser
        ai_result = await websocket.recv()
        print(f"Received result from Gemini Nano: {ai_result}")

    except websockets.ConnectionClosed:
        print("Browser connection closed.")

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())

JavaScript Client (bridge_client.html - Conceptual):

code
Html
play_circle
download
content_copy
expand_less
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

Disclaimer: This is a highly experimental project. The features and file locations we are targeting are internal to Chrome and are not guaranteed to be stable. They can change or be removed at any time. This project is not affiliated with or endorsed by Google.
