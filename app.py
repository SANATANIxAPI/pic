import os
import io
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import cv2
import numpy as np
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Configuration
API_ID = 28795512  # Replace with your Telegram API ID
API_HASH = "c17e4eb6d994c9892b8a8b6bfea4042a"  # Replace with your Telegram API HASH
BOT_TOKEN = "8028016578:AAFNV8_PxVqB_98fALDTo98V-8x0NhQIlzc"  # Replace with your bot token
HOST = "0.0.0.0"
PORT = 8000

# Initialize FastAPI
app = FastAPI(title="Image Enhancer API + Bot")

# Initialize Pyrogram
bot = Client(
    "image_enhancer_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

# Global variables to store user sessions
user_sessions = {}

# Initialize models
def init_models():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path="weights/RealESRGAN_x4plus.pth",
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True
    )
    return upsampler

upsampler = init_models()

# Image processing functions
def apply_enhancement(image: np.ndarray, quality: str):
    if quality == "low":
        enhanced = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    elif quality == "medium":
        enhanced = cv2.detailEnhance(image, sigma_s=5, sigma_r=0.15)
    elif quality == "high":
        enhanced = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
        enhanced = cv2.edgePreservingFilter(enhanced, flags=1, sigma_s=64, sigma_r=0.2)
    elif quality == "ultra":
        enhanced = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        enhanced = cv2.detailEnhance(enhanced, sigma_s=15, sigma_r=0.2)
    elif quality == "4k":
        enhanced, _ = upsampler.enhance(image, outscale=4)
    else:
        enhanced = image
    
    return enhanced

# FastAPI Endpoints
@app.post("/api/enhance")
async def api_enhance_image(
    file: UploadFile = File(...),
    quality: str = "high",
    output_format: str = "jpg"
):
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        enhanced = apply_enhancement(image, quality)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(enhanced_rgb)
        
        output_bytes = io.BytesIO()
        result_image.save(output_bytes, format=output_format, quality=95)
        output_bytes.seek(0)
        
        return StreamingResponse(output_bytes, media_type=f"image/{output_format}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Telegram Bot Handlers
@bot.on_message(filters.photo & filters.private)
async def handle_photo(client, message):
    user_id = message.from_user.id
    file_id = message.photo.file_id
    
    # Download the photo
    file_path = await bot.download_media(file_id, file_name=f"temp_{user_id}.jpg")
    
    # Store in user session
    user_sessions[user_id] = {"file_path": file_path}
    
    # Ask for quality
    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Low", callback_data="quality_low"),
             InlineKeyboardButton("Medium", callback_data="quality_medium")],
            [InlineKeyboardButton("High", callback_data="quality_high"),
             InlineKeyboardButton("Ultra", callback_data="quality_ultra")],
            [InlineKeyboardButton("4K Upscale", callback_data="quality_4k")]
        ]
    )
    
    await message.reply_text(
        "Please select enhancement quality:",
        reply_markup=keyboard
    )

@bot.on_callback_query()
async def handle_quality_selection(client, callback_query):
    user_id = callback_query.from_user.id
    quality = callback_query.data.replace("quality_", "")
    
    if user_id not in user_sessions:
        await callback_query.answer("Session expired. Please send the image again.")
        return
    
    file_path = user_sessions[user_id]["file_path"]
    
    # Process the image
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"http://localhost:{PORT}/api/enhance",
                files=files,
                params={"quality": quality}
            )
        
        if response.status_code == 200:
            output_bytes = io.BytesIO(response.content)
            output_bytes.name = f"enhanced_{quality}.jpg"
            await bot.send_photo(
                chat_id=user_id,
                photo=output_bytes,
                caption=f"Here's your {quality} quality enhanced image!"
            )
        else:
            await bot.send_message(
                chat_id=user_id,
                text="Sorry, there was an error processing your image."
            )
    
    except Exception as e:
        await bot.send_message(
            chat_id=user_id,
            text=f"Error: {str(e)}"
        )
    
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)
        if user_id in user_sessions:
            del user_sessions[user_id]
        
    await callback_query.answer()

# Startup event
@app.on_event("startup")
async def startup_event():
    await bot.start()
    print("Bot started!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    await bot.stop()
    print("Bot stopped!")

# Run the application
if __name__ == "__main__":
    import uvicorn
    from threading import Thread
    
    # Create weights directory if not exists
    os.makedirs("weights", exist_ok=True)
    
    # Start FastAPI server in a separate thread
    def run_fastapi():
        uvicorn.run(app, host=HOST, port=PORT)
    
    fastapi_thread = Thread(target=run_fastapi)
    fastapi_thread.start()
    
    # Run Pyrogram bot in main thread
    print("Starting bot...")
    bot.run()
