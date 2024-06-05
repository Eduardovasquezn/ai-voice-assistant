# AI-Voice-Assistant
Welcome to the AI Voice Assistant project! This assistant can start conversations, transcribe audio to text, 
generate responses, and convert text back to speech, all while showcasing a sleek frontend interface. It serves as a 
complement to the concepts discussed in the accompanying [YouTube video](https://youtu.be/OqoNkqAsl2Q), offering a 
practical implementation  of the discussed techniques.

## Features
- Conversation Starter: Begin a conversation with the AI assistant.
- Audio to Text: Transcribe audio to text using an OpenAI model.
- Fast Inference: Generate responses quickly with Groq.
- Text to Speech: Convert text back to speech using gTTS.
- Frontend Design: Design the frontend interface using Streamlit.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Eduardovasquezn/ai-voice-assistant.git
   
2. Navigate to the project directory:
    ```bash
    cd ai-voice-assistant
    ```
3. Create and activate virtual environment:
    ```bash
    python -m venv venv
    venv/Scripts/activate
    ```
4. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1. Create a `.env` file using `.env-example` as a template:
    ```bash
    cp .env-example .env
     ```
2. Run the main application script:
    ```bash
    streamlit run src/app.py
    ```
   

### Learn More
 
Don't forget to check out the video, like, comment, and subscribe for more advanced tutorials!

If you found the content helpful, consider subscribing to my 
[YouTube channel](https://www.youtube.com/channel/UCYZ_si4TG801SAuLrNl-v-g?sub_confirmation=1) to support me.
