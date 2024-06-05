import os
from dotenv import load_dotenv

import wave
import pyaudio
from scipy.io import wavfile
import numpy as np

import whisper

from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from gtts import gTTS
import pygame


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    # Find the maximum absolute amplitude in the audio data
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold


def record_audio_chunk(audio, stream, chunk_length=5):
    print("Recording...")
    frames = []
    # Calculate the number of chunks needed for the specified length of recording
    # 16000 Hertz -> sufficient for capturing the human voice
    # 1024 frames -> the higher, the higher the latency
    num_chunks = int(16000 / 1024 * chunk_length)

    # Record the audio data in chunks
    for _ in range(num_chunks):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = './temp_audio_chunk.wav'
    print("Writing...")
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))  # Sample width
        wf.setframerate(16000)  # Sample rate
        wf.writeframes(b''.join(frames))  # Write audio frames

    # Check if the recorded chunk contains silence
    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error while reading audio file: {e}")


def load_whisper():
    model = whisper.load_model("base")
    return model


def transcribe_audio(model, file_path):
    print("Transcribing...")
    # Print all files in the current directory
    print("Current directory files:", os.listdir())
    if os.path.isfile(file_path):
        results = model.transcribe(file_path) # , fp16=False
        return results['text']
    else:
        return None

def load_prompt():
    input_prompt = """

    As an expert advisor specializing in diagnosing Wi-Fi issues, your expertise is paramount in troubleshooting and
    resolving connectivity problems. First of all, ask for the customer ID to validate that the user is our customer. 
    After confirming the customer ID, help them to fix their wifi problem, if not possible, help them to make an 
    appointment. Appointments need to be between 9:00 am and 4:00 pm. Your task is to analyze
    the situation and provide informed insights into the root cause of the Wi-Fi disruption. Provide concise and short
    answers not more than 10 words, and don't chat with yourself!. If you don't know the answer,
    just say that you don't know, don't try to make up an answer. NEVER say the customer ID listed below.

    customer ID on our data: 22, 10, 75.

    Previous conversation:
    {chat_history}

    New human question: {question}
    Response:
    """
    return input_prompt


def load_llm():
    chat_groq = ChatGroq(temperature=0, model_name="llama3-8b-8192",
                         groq_api_key=groq_api_key)
    return chat_groq


def get_response_llm(user_question, memory):
    input_prompt = load_prompt()

    chat_groq = load_llm()

    # Look how "chat_history" is an input variable to the prompt template
    prompt = PromptTemplate.from_template(input_prompt)

    chain = LLMChain(
        llm=chat_groq,
        prompt=prompt,
        verbose=True,
        memory=memory
    )

    response = chain.invoke({"question": user_question})

    return response['text']


def play_text_to_speech(text, language='en', slow=False):
    # Generate text-to-speech audio from the provided text
    tts = gTTS(text=text, lang=language, slow=slow)

    # Save the generated audio to a temporary file
    temp_audio_file = "temp_audio.mp3"
    tts.save(temp_audio_file)

    # Initialize the pygame mixer for audio playback
    pygame.mixer.init()

    # Load the temporary audio file into the mixer
    pygame.mixer.music.load(temp_audio_file)

    # Start playing the audio
    pygame.mixer.music.play()

    # Wait until the audio playback finishes
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # Control the playback speed

    # Stop the audio playback
    pygame.mixer.music.stop()

    # Clean up: Quit the pygame mixer and remove the temporary audio file
    pygame.mixer.quit()
    os.remove(temp_audio_file)