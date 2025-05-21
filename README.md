# LoRA Fine-Tuning for Stable Diffusion 🚀
In this project the ChatGPT is in good Amount, but **this is not just a copy-paste project**.  

### 🌱 How It Started  
Initially, my goal was **just to generate images** using different models. I worked with **ComfyUI** and various **Stable Diffusion models**. My first attempt was with **Juggernaut**, but it **crashed my laptop** instantly. That’s when I **started learning**—about **SDXL, SD 1.5, SD 3, and LoRA fine-tuning**—and I optimized different models for my hardware.  

### 🤖 Why LoRA?  
After experimenting, I realized that **fine-tuning a LoRA model** would be the best way to **make Stable Diffusion work efficiently on my laptop**. That’s when I decided to ask ChatGPT to assist me in structuring the fine-tuning process.

I’m being **honest** about this because I don’t just want to **copy-paste** code—I want to **learn** and truly understand the process. 

---

## 📂 Project Overview  

✅ **Fine-tunes Stable Diffusion using LoRA** on a custom dataset  
✅ **Optimized for low-VRAM (even 4GB GPUs can handle it!)**  
✅ **ComfyUI workflow integration** for real-time inference  
✅ **Supports Hugging Face, xFormers, and other optimizations**  

---

## 🛠️ Installation  

```bash
# Clone the repository
git clone https://github.com/arculus07/LoRA-Fine-Tuning-Stable-Diffusion.git
cd LoRA-Fine-Tuning-Stable-Diffusion

# Install dependencies
pip install -r requirements.txt

python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

git clone https://github.com/comfyanonymous/ComfyUI.git

python main.py  # Inside ComfyUI folder
