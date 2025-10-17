# Project_FreeGeminiNano-A_Python_Bridge_for_Chrome-s_On-Device_Gemini_Nano
An ambitious open-source initiative to unlock the full potential of Google's Gemini Nano model by making it directly accessible to Python developers. We aim to bridge the gap between the model's secure, sandboxed environment within Chrome and the powerful, flexible world of Python and its machine learning ecosystem.

The goal is to break the model out of its sandbox for greater flexibility and research, and extension such as implementing cutting-edge techniques like **Artificial Hippocampus Networks** (as described in [arXiv:2510.07318](https://arxiv.org/pdf/2510.07318)) to efficiently manage and expand the model's effective context length, overcoming the current token limits.  See e.g., 
"Learn Artificial Hippocampus Networks (AHN) with a Pure PyTorch Qwen2 Model" at:
https://huggingface.co/MartialTerran/Toy_Qwen2-AHN_ByteDance-Seed_AHN/blob/main/README.md

https://huggingface.co/MartialTerran/Toy_Qwen2-AHN_ByteDance-Seed_AHN/edit/main/Qwen2-AHN_model_v1.0.1_Gemini2.5_fixed.py 

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


Here is a README.md and project title that captures the spirit of this endeavor.

Project Heimdall: A Python Bridge for Gemini Nano

Our mission: To unlock Chrome's on-device Gemini Nano model, making its weights and architecture directly accessible from Python for advanced research and application development.

This project is named Heimdall, after the Bifröst bridge guardian, because our goal is to create a bridge between two distinct realms: the sandboxed, high-performance environment of the Chrome browser and the flexible, powerful machine learning ecosystem of Python.

The Vision: Supercharging On-Device AI

Google has integrated a powerful version of its Gemini model, called Gemini Nano, directly into the Chrome browser. This model runs entirely locally, offering unparalleled speed, privacy, and zero API costs. However, it is currently only accessible through a sandboxed JavaScript API (window.ai).

Project Heimdall aims to break down this barrier. We have two primary objectives:

(The Ultimate Goal) Direct Python Inference: To reverse-engineer the Gemini Nano model format and its surrounding logic, allowing the model files (weights.bin) to be loaded and run directly within a Python environment (e.g., using PyTorch or TensorFlow).

(The Pragmatic Bridge) Python-to-WebGPU Bridging: To create a seamless, high-performance bridge where Python scripts can send prompts to a headless Chrome instance, have the browser execute the model on the GPU via WebGPU, and stream the results back to Python.

Key Research Goal: Extending Gemini Nano's Context Window

Directly accessing the model's weights and architecture in Python would be a game-changer for the open-source AI community. It would allow us to experiment with and extend the model's core capabilities.

A primary research goal of this project is to implement techniques for efficient long-context modeling. For example, we aim to integrate an Artificial Hippocampus Network (AHN), as described in the paper "Artificial Hippocampus Networks for Efficient Long-Context Modeling" and prototyped in Martial Terran's educational Qwen2-AHN implementation.

By "jailbreaking" the model, we could potentially graft or integrate AHN-style layers onto the Gemini Nano graph, allowing it to handle vastly longer context windows than its default configuration, all while maintaining its on-device efficiency.

The Technical Hurdles: Why This Is Hard

The files for Gemini Nano, located in Chrome's protected user data directory, are not standard model weights. To use them, we must overcome several significant challenges:

Reconstructing Input Preprocessing: The model requires a precisely formatted numerical tensor as input. We must reverse-engineer the logic Chrome uses to convert raw images, audio, and text into this format, including:

Exact image dimensions and resizing methods.

Color channel ordering (RGB vs. BGR).

Pixel value normalization schemes.

Reconstructing Output Postprocessing: The model's output is a tensor of raw probabilities (logits), not human-readable text. We need to reconstruct the entire decoding pipeline that turns these numbers into words. This involves understanding how to use auxiliary files like:

.binarypb (Protocol Buffers) for graph definitions.

.fst (Finite State Transducers) and .syms for language modeling and constraining output.

This logic is currently embedded within Chrome's internal C++ source code.

How You Can Contribute

This is a complex reverse-engineering project, and we need your help! We're looking for contributors with the following skills:

ML Reverse Engineers: Individuals experienced in analyzing and converting proprietary model formats.

Chromium Source Code Experts: Developers comfortable navigating the Chromium C++ codebase to find the pre/post-processing logic.

TensorFlow / PyTorch / ONNX Specialists: Engineers who can help reconstruct the model graph and weight-loading mechanisms in Python.

WebGPU & Browser Automation Experts: For our pragmatic bridging goal, we need developers who can build a high-performance Python-to-JavaScript communication layer using tools like Playwright and WebSockets.

Project Roadmap & First Steps
Phase 1: The WebGPU Bridge (Track 2)

Develop a stable WebSocket-based server in Python.

Create a minimal HTML/JS client that can receive prompts, execute them via window.ai, and stream results back.

Bundle this into a user-friendly Python library.

Phase 2: The Direct Inference Engine (Track 1)

Locate and Analyze Preprocessing Code: Dive into the Chromium source to find the functions that prepare data for the model.

Reverse-Engineer the weights.bin Format: Analyze the binary format of the model weights and graph.

Rebuild the Decoder: Re-implement the output postprocessing logic in Python to turn logits into text.

Build a Python Loader: Create a Python class capable of loading the weights and running an inference step.

Getting Started: A Python Script to Find the Model

This script will help you locate and copy the Gemini Nano model files from your local Chrome installation. This is the first step for any analysis.

find_nano.py

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

Disclaimer: This is an independent, experimental project for educational and research purposes. The internal APIs and model formats of Chrome are undocumented, subject to change without notice, and are not intended for external use. This project is not affiliated with or endorsed by Google.
