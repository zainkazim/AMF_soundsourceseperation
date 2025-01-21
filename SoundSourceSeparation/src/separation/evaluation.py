import os
import scipy.io.wavfile as wav
import mir_eval
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt

def convert_alaw_to_pcm(input_file, output_file):
    """
    Converts an ALAW encoded WAV file to PCM 16-bit signed little-endian format.
    """
    audio = AudioSegment.from_wav(input_file)
    audio.export(output_file, format='wav', codec='pcm_s16le')

# Paths to reference and separated files
reference_dir = '/home/ZA/Desktop/SoundSoulSeparationViaBins/output/0_channel.wav'  # Directory containing clean reference files
separated_dir = '/home/ZA/Desktop/SoundSoulSeparationViaBins/output/1_channel.wav'  # Directory containing separated output files

# List of channels (for example, if you have 2 channels in the reference file)
channels = [0, 1]  # Update this list based on the number of channels in your reference audio

# Initialize lists to store metrics
sdr_list = []
sir_list = []
sar_list = []

# Iterate through each channel and compute the metrics
for mic_index in channels:
    # Paths for reference and separated files for this channel
    reference_file = os.path.join(reference_dir, f"channel.wav")  # Reference file pattern
    separated_file = os.path.join(separated_dir, f"{mic_index}_channel.wav")  # Separated file pattern
    
    # Convert reference file from ALAW to PCM if necessary
    converted_reference_file = os.path.join(reference_dir, f"converted_channel_{mic_index}.wav")
    if reference_file.lower().endswith(".wav"):
        try:
            # Convert to PCM if necessary
            convert_alaw_to_pcm(reference_file, converted_reference_file)
            reference_file = converted_reference_file  # Update reference file to the converted one
            print(f"Converted {reference_file} to PCM format and saved to {converted_reference_file}")
        except Exception as e:
            print(f"Error converting {reference_file}: {e}")

    # Check if the separated file exists
    if not os.path.exists(separated_file):
        print(f"Warning: Separated file {separated_file} does not exist.")
        continue

    # Read the reference and separated audio files
    rate_ref, reference_audio = wav.read(reference_file)
    rate_sep, separated_audio = wav.read(separated_file)

    # Debugging information to check the lengths
    print(f"Sample rate of reference audio: {rate_ref}")
    print(f"Sample rate of separated audio: {rate_sep}")
    print(f"Length of reference audio: {len(reference_audio)} samples")
    print(f"Length of separated audio: {len(separated_audio)} samples")

    # Ensure both signals are mono and of the same length
    if len(reference_audio) != len(separated_audio):
        print(f"Trimming or padding the separated audio to match the reference audio length...")
        # Trim or pad the separated audio to match the length of the reference audio
        if len(separated_audio) > len(reference_audio):
            separated_audio = separated_audio[:len(reference_audio)]  # Trim
        else:
            # Pad with zeros if separated audio is shorter
            padding = np.zeros((len(reference_audio) - len(separated_audio), separated_audio.shape[1]))
            separated_audio = np.concatenate((separated_audio, padding), axis=0)

    # If stereo, take the corresponding channel from reference (for example, channel 0 or 1)
    if len(reference_audio.shape) > 1:
        reference_audio = reference_audio[:, mic_index]  # Select the corresponding channel

    # If the separated audio is stereo, you might need to extract one channel
    if len(separated_audio.shape) > 1:
        separated_audio = separated_audio[:, 0]  # Assuming separated is mono for each source

    # Calculate the metrics (SDR, SIR, SAR) for this channel
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(reference_audio, separated_audio)

    # Handle inf values in SIR (setting them to a large value for display)
    if np.isinf(sir[0]):
        sir[0] = 100  # Set inf values to 100 dB for better visualization
    
    # Store the results
    sdr_list.append(sdr[0])  # Access the first SDR value
    sir_list.append(sir[0])  # Access the first SIR value
    sar_list.append(sar[0])  # Access the first SAR value

    # Print the metrics for the current channel
    print(f"Channel {mic_index}:")
    print(f"  SDR (Signal-to-Distortion Ratio): {sdr[0]:.2f} dB")  # Print the first SDR value
    print(f"  SIR (Signal-to-Interference Ratio): {sir[0]:.2f} dB")  # Print the first SIR value
    print(f"  SAR (Signal-to-Artifacts Ratio): {sar[0]:.2f} dB")  # Print the first SAR value
    print()

# Optionally, calculate the average of the metrics across all channels
if sdr_list and sir_list and sar_list:  # Ensure there are metrics to average
    avg_sdr = sum(sdr_list) / len(sdr_list)
    avg_sir = sum(sir_list) / len(sir_list)
    avg_sar = sum(sar_list) / len(sar_list)

    print(f"Average SDR: {avg_sdr:.2f} dB")
    print(f"Average SIR: {avg_sir:.2f} dB")
    print(f"Average SAR: {avg_sar:.2f} dB")
else:
    print("No valid channels to evaluate.")

# Plot the results
# Create an array to hold the x positions for each bar
x = np.arange(len(channels))

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the values
bar_width = 0.25  # Width of each bar
ax.bar(x - bar_width, sdr_list, width=bar_width, label='SDR (dB)', color='b')
ax.bar(x, sar_list, width=bar_width, label='SAR (dB)', color='r')
ax.bar(x + bar_width, sir_list, width=bar_width, label='SIR (dB)', color='g')

# Set labels and title
ax.set_xlabel('Channels')
ax.set_ylabel('Value (dB)')
ax.set_title('Performance of Sound Separation for Each Channel')
ax.set_xticks(x)
ax.set_xticklabels([f'Channel {i}' for i in channels])

# Adjust the y-axis for better visualization of inf values
ax.set_ylim(-20, 120)  # Adjust y-limit to show all values clearly

# Add legend
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
