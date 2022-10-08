
import multiprocessing
import numpy as np # required to avoid crashing in assigning the callback input which is a numpy object
import os
import sounddevice as sd
import string
import subprocess
import sys
import time as t
import wavio
import webrtcvad
import whisper

whisper_model = whisper.load_model("base")

ACTIVATION_PHRASE = "hello whisper"

# Parts taken from https://github.com/wiseman/py-webrtcvad/issues/29#issuecomment-627403563

channels = [1]
# translate channel numbers to be 0-indexed
mapping  = [c - 1 for c in channels]

# get the default audio input device and its sample rate
device_info = sd.query_devices(None, 'input')
sample_rate = int(device_info['default_samplerate'])

DURATION = 30
total_frames = DURATION * sample_rate

interval_size = 30 # audio interval size in ms
downsample = 1

block_size = sample_rate * interval_size / 1000

# get an instance of webrtc's voice activity detection
vad = webrtcvad.Vad()

output = np.empty((total_frames, channels[0]), "float32")
has_command_started = False
num_voice_blocks = 0
num_silent_blocks_after_voice_activity = 0
frame = 0
total_blocks_processed = 0

result = {"text": "Starting text."}

def voice_activity_detection(audio_data):
    return vad.is_speech(audio_data, sample_rate)


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    global frame
    global total_blocks_processed
    global num_voice_blocks
    global num_silent_blocks_after_voice_activity

    if status:
        print(F"underlying audio stack warning:{status}", file=sys.stderr)

    assert frames == block_size
    audio_data = indata[::downsample, mapping]        # possibly downsample, in a naive way

    if frame + int(block_size) > total_frames:  # Begin overwriting old audio data.
        frame = 0
    output[frame:frame + int(block_size), mapping] = audio_data  # Store unprocessed audio data to output ndarray for file storage.
    frame = frame + int(block_size)
    total_blocks_processed += 1

    is_running = False
    if num_silent_blocks_after_voice_activity > 67 and total_blocks_processed > 100:
        # Reset if no voice for 2 seconds
        num_silent_blocks_after_voice_activity = 0
        num_voice_blocks = 0

        for p in multiprocessing.active_children():
            if p.name == "whisperingdiffusion":
                is_running = True
        if not is_running:
            print("Storing audio file...")
            temp_output = np.roll(output, -1 * frame)  # Re-order the output array to preserve chronology
            audio_file_path = "whisper/Potential Recording.wav"
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            wavio.write(audio_file_path, temp_output, sample_rate, sampwidth=1)
            process = multiprocessing.Process(target=call_model, name="whisperingdiffusion", args=(audio_file_path,))
            process.start()

    audio_data = map(lambda x: (x+1)/2, audio_data)   # normalize from [-1,+1] to [0,1], you might not need it with different microphones/drivers
    audio_data = np.fromiter(audio_data, np.float16)  # adapt to expected float type

    audio_data = audio_data.tobytes()
    detection = voice_activity_detection(audio_data)

    if detection and total_blocks_processed > 10:
        num_voice_blocks += 1
        num_silent_blocks_after_voice_activity = 0
    elif num_voice_blocks > 50:  # If there's no current voice detection, but in the past we have heard voice for at least 1.5 seconds:
        num_silent_blocks_after_voice_activity += 1
    print(f'{detection} \r', end="") # use just one line to show the detection status (speech / not-speech)

def call_whisper(audio_file_path, queue):
    print(f"Beginning transcription process...")
    transcription = whisper_model.transcribe(audio_file_path, best_of=5)["text"]
    queue.put(transcription)

def call_model(audio_file_path):
    before = t.time()

    # All of this to avoid long transcriptions of empty noise
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=call_whisper, name="whispercall", args=(audio_file_path, queue))
    process.start()
    while process.is_alive():
        if t.time() - before > 10:
            process.kill()
            print("Oh no, the transcription took too long! Killing process...")
            return
    transcription = queue.get()

    # Clean the text, search for activation phrase. If found, generate image with Stable Diffusion
    cleaned_text: str = transcription.lower().translate(str.maketrans('', '', string.punctuation))
    print(cleaned_text)
    try:
        command = cleaned_text[cleaned_text.rindex(ACTIVATION_PHRASE) + len(ACTIVATION_PHRASE):]
        if command.startswith("r "):  # sometimes it thinks you're saying "hello whisperr"
            command = command[2:]
        if command.startswith("er "):  # sometimes it thinks you're saying "hello whisperr"
            command = command[3:]
        os.rename(audio_file_path, f"whisper/{command}.wav")
        print(f"Command: {command}")
        subprocess.run(["python", "scripts/txt2img.py", "--fixed_code", "--ddim_steps", "12", "--prompt", f"\"{command}\""])
    except Exception as e:
        pass
    print(f"Process time: {t.time() - before}")

if __name__ == "__main__":
    print("reading audio stream from default audio input device:\n" + str(sd.query_devices()) + '\n')
    print(F"audio input channels to process: {channels}")
    print(F"sample_rate: {sample_rate}")
    print(F"window size: {interval_size} ms")
    print(F"datums per window: {block_size}")
    print()

    with sd.InputStream(
        device=None,  # the default input device
        channels=max(channels),
        samplerate=sample_rate,
        blocksize=int(block_size),
        callback=audio_callback):

        t.sleep(DURATION * 30)



