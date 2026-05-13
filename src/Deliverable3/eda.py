import os
import glob
import random
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from collections import Counter
from load_data import load_and_format, TARGET_SR, DURATION
from dataset import AudioNoiseDataset
from main import get_splits
from torch.utils.data import DataLoader

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def download_datasets():
    #Download datasets using KaggleHub
    print("="*60)
    print("DOWNLOADING DATASETS")
    print("="*60)
    
    print("\nDownloading LibriSpeech ASR Corpus (Clean Speech)...")
    clean_dir = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
    print(f" Clean speech dataset located at: {clean_dir}")
    
    print("\nDownloading Environmental Sound Classification 50 (Noise)...")
    noise_dir = kagglehub.dataset_download("mmoreaux/environmental-sound-classification-50")
    print(f" Noise dataset located at: {noise_dir}")
    
    return clean_dir, noise_dir

def get_all_audio_files(directory):
    #Get all audio files from directory
    wav_files = glob.glob(os.path.join(directory, '**', '*.wav'), recursive=True)
    flac_files = glob.glob(os.path.join(directory, '**', '*.flac'), recursive=True)
    return wav_files + flac_files

def analyze_file_distribution(clean_dir, noise_dir):
    #Analyze the distribution of audio files
    print(f"\n{'='*60}")
    print("FILE DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    
    clean_files = get_all_audio_files(clean_dir)
    noise_files = get_all_audio_files(noise_dir)
    
    # Count file extensions
    clean_extensions = Counter([os.path.splitext(f)[1].lower() for f in clean_files])
    noise_extensions = Counter([os.path.splitext(f)[1].lower() for f in noise_files])
    
    print(f"\nClean Speech Files: {len(clean_files)}")
    for ext, count in clean_extensions.items():
        print(f"  {ext}: {count} files ({count/len(clean_files)*100:.1f}%)")
    
    print(f"\nNoise Files: {len(noise_files)}")
    for ext, count in noise_extensions.items():
        print(f"  {ext}: {count} files ({count/len(noise_files)*100:.1f}%)")
    
    return clean_files, noise_files

def analyze_audio_durations(clean_files, noise_files, num_samples=500):
    #Analyze the duration distribution of audio files
    print(f"\n{'='*60}")
    print("AUDIO DURATION ANALYSIS")
    print(f"{'='*60}")
    
    # Sample files for faster analysis
    clean_sample = random.sample(clean_files, min(num_samples, len(clean_files)))
    noise_sample = random.sample(noise_files, min(num_samples, len(noise_files)))
    
    clean_durations = []
    noise_durations = []
    
    print("\nAnalyzing clean speech durations...")
    for i, file in enumerate(clean_sample):
        try:
            duration = librosa.get_duration(path=file)
            clean_durations.append(duration)
        except:
            pass
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(clean_sample)} files...")
    
    print("\nAnalyzing noise durations...")
    for i, file in enumerate(noise_sample):
        try:
            duration = librosa.get_duration(path=file)
            noise_durations.append(duration)
        except:
            pass
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(noise_sample)} files...")
    
    # Statistics
    if clean_durations:
        print(f"\nClean Speech Duration Statistics:")
        print(f"  Min:    {np.min(clean_durations):.2f}s")
        print(f"  Max:    {np.max(clean_durations):.2f}s")
        print(f"  Mean:   {np.mean(clean_durations):.2f}s")
        print(f"  Median: {np.median(clean_durations):.2f}s")
        print(f"  Std:    {np.std(clean_durations):.2f}s")
    
    if noise_durations:
        print(f"\nNoise Duration Statistics:")
        print(f"  Min:    {np.min(noise_durations):.2f}s")
        print(f"  Max:    {np.max(noise_durations):.2f}s")
        print(f"  Mean:   {np.mean(noise_durations):.2f}s")
        print(f"  Median: {np.median(noise_durations):.2f}s")
        print(f"  Std:    {np.std(noise_durations):.2f}s")
    
    return clean_durations, noise_durations

def plot_duration_distributions(clean_durations, noise_durations, save_dir="src/Deliverable3/eda_images"):
    #Plot duration distributions
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Clean speech durations
    if clean_durations:
        axes[0].hist(clean_durations, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(DURATION, color='red', linestyle='--', linewidth=2, label=f'Target: {DURATION}s')
    axes[0].set_xlabel('Duration')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Clean Speech Duration Distribution')
    axes[0].legend()
    
    # Noise durations
    if noise_durations:
        axes[1].hist(noise_durations, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1].axvline(DURATION, color='red', linestyle='--', linewidth=2, label=f'Target: {DURATION}s')
    axes[1].set_xlabel('Duration')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Noise Duration Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'duration_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Duration distribution plot saved to {save_dir}/duration_distribution.png")

def analyze_train_val_test_split(clean_dir, noise_dir):
    """Analyze the train/validation/test split."""
    print(f"\n{'='*60}")
    print("TRAIN/VALIDATION/TEST SPLIT ANALYSIS")
    print(f"{'='*60}")
    
    # Get splits
    clean_train, clean_val, clean_test = get_splits(clean_dir)
    noise_train, noise_val, noise_test = get_splits(noise_dir)
    
    total_clean = len(clean_train) + len(clean_val) + len(clean_test)
    total_noise = len(noise_train) + len(noise_val) + len(noise_test)
    
    print(f"\nClean Speech Split:")
    print(f"  Train:      {len(clean_train)} files ({len(clean_train)/total_clean*100:.1f}%)")
    print(f"  Validation: {len(clean_val)} files ({len(clean_val)/total_clean*100:.1f}%)")
    print(f"  Test:       {len(clean_test)} files ({len(clean_test)/total_clean*100:.1f}%)")
    print(f"  Total:      {total_clean} files")
    
    print(f"\nNoise Split:")
    print(f"  Train:      {len(noise_train)} files ({len(noise_train)/total_noise*100:.1f}%)")
    print(f"  Validation: {len(noise_val)} files ({len(noise_val)/total_noise*100:.1f}%)")
    print(f"  Test:       {len(noise_test)} files ({len(noise_test)/total_noise*100:.1f}%)")
    print(f"  Total:      {total_noise} files")
    
    # Verify no overlap
    train_val_overlap_clean = set(clean_train) & set(clean_val)
    train_test_overlap_clean = set(clean_train) & set(clean_test)
    val_test_overlap_clean = set(clean_val) & set(clean_test)
    
    
    return clean_train, clean_val, clean_test, noise_train, noise_val, noise_test

def plot_split_distribution(clean_train, clean_val, clean_test, 
                            noise_train, noise_val, noise_test,
                            save_dir="src/Deliverable3/eda_images"):
    #Plot the train/val/test split distribution
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Clean speech
    clean_counts = [len(clean_train), len(clean_val), len(clean_test)]
    clean_labels = ['Train', 'Validation', 'Test']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    axes[0].bar(clean_labels, clean_counts, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Number of Files')
    axes[0].set_title('Clean Speech Split Distribution')
    for i, count in enumerate(clean_counts):
        axes[0].text(i, count + 50, str(count), ha='center', fontweight='bold')
    
    # Noise
    noise_counts = [len(noise_train), len(noise_val), len(noise_test)]
    
    axes[1].bar(clean_labels, noise_counts, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('Number of Files')
    axes[1].set_title('Noise Split Distribution')
    for i, count in enumerate(noise_counts):
        axes[1].text(i, count + 50, str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'split_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Split distribution plot saved to {save_dir}/split_distribution.png")

def analyze_snr_effect(clean_files, noise_files, save_dir="src/Deliverable3/eda_images"):
    #Demonstrate the effect of different SNR levels on audio mixing
    os.makedirs(save_dir, exist_ok=True)
    
    clean_audio = load_and_format(random.choice(clean_files))
    noise_audio = load_and_format(random.choice(noise_files))
    
    p_clean = np.mean(clean_audio ** 2)
    p_noise = np.mean(noise_audio ** 2)
    
    snr_levels = [-5, 0, 5, 10, 15]
    
    fig, axes = plt.subplots(len(snr_levels), 1, figsize=(14, 12))
    
    for i, snr in enumerate(snr_levels):
        scalar = np.sqrt(p_clean / (10**(snr/10) * p_noise))
        noisy = clean_audio + (scalar * noise_audio[:len(clean_audio)])
        
        # Normalize
        max_val = np.max(np.abs(noisy))
        if max_val > 1.0:
            noisy = noisy / max_val
        
        time = np.arange(len(noisy)) / TARGET_SR
        axes[i].plot(time, noisy, color='red', alpha=0.7)
        axes[i].set_ylabel('Amplitude')
        axes[i].set_title(f'SNR = {snr} dB')
        axes[i].set_xlim([0, min(2.0, DURATION)])  # Show first 2 seconds
    
    axes[-1].set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'snr_effect.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SNR effect plot saved to {save_dir}/snr_effect.png")

def print_audio_means(clean_files, noise_files, num_samples=100):
    print(f"\n{'='*60}")
    print("AUDIO MEAN ANALYSIS")
    print(f"{'='*60}")
    
    #Samples for fastest analysis
    clean_sample = random.sample(clean_files, min(num_samples, len(clean_files)))
    noise_sample = random.sample(noise_files, min(num_samples, len(noise_files)))
    
    clean_means = []
    noise_means = []
    
    print(f"\nCalculating the means of{len(clean_sample)} clean audios...")
    for i, file in enumerate(clean_sample):
        try:
            audio = load_and_format(file)
            mean_val = np.mean(audio)
            clean_means.append(mean_val)
            if (i + 1) % 50 == 0:
                print(f"  Procesed {i+1}/{len(clean_sample)}")
        except Exception as e:
            print(f"  Error procesando {file}: {e}")
    
    print(f"\nCalculating the means of {len(noise_sample)} noise")
    for i, file in enumerate(noise_sample):
        try:
            audio = load_and_format(file)
            mean_val = np.mean(audio)
            noise_means.append(mean_val)
            if (i + 1) % 50 == 0:
                print(f"  Procesed {i+1}/{len(noise_sample)}")
        except Exception as e:
            print(f"  Error {file}: {e}")
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    if clean_means:
        print(f"\nCLEANES AUDIOS:")
        print(f"   Mean: {np.mean(clean_means):.6f}")
        print(f"   Standar desviation: {np.std(clean_means):.6f}")
        print(f"   Minimal value: {np.min(clean_means):.6f}")
        print(f"   Maximal value: {np.max(clean_means):.6f}")
        print(f"   Median: {np.median(clean_means):.6f}")
    
    if noise_means:
        print(f"\nNOISE AUDIO:")
        print(f"   Mean: {np.mean(noise_means):.6f}")
        print(f"   Standar desviation: {np.std(noise_means):.6f}")
        print(f"   Minimal value: {np.min(noise_means):.6f}")
        print(f"   Maximal value: {np.max(noise_means):.6f}")
        print(f"   Median: {np.median(noise_means):.6f}")

    # comparisons
    if clean_means and noise_means:
        print(f"\nCOMPARISONS:")
        print(f"   Means difference: {abs(np.mean(clean_means) - np.mean(noise_means)):.6f}")
        if np.mean(clean_means) > np.mean(noise_means):
            print(f"   The cleaned audio have higher mean amplitude than the noisy audio")
        else:
            print(f"   The noisy audio have higher mean amplitude than the cleaned audio")
    
    print(f"\n{'='*60}")
    
    # visualization of the mean distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    if clean_means:
        ax.hist(clean_means, bins=30, alpha=0.7, label='Clean Audio', color='blue')
    if noise_means:
        ax.hist(noise_means, bins=30, alpha=0.7, label='Noise', color='orange')
    ax.set_xlabel('Average amplitude')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Audio Means')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('src/Deliverable3/eda_images/audio_means_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Graph saved in: src/Deliverable3/eda_images/audio_means_distribution.png")
    
    return clean_means, noise_means

def generate_summary_report(clean_dir, noise_dir, clean_files, noise_files,
                            clean_durations, noise_durations,
                            clean_train, clean_val, clean_test,
                            noise_train, noise_val, noise_test):
    """Generate a summary report of the EDA."""
    print(f"\n{'='*60}")
    print("EDA SUMMARY REPORT")
    print(f"{'='*60}")
    
    print(f"""
    Dataset Overview:
    -----------------
    Clean Speech Source: LibriSpeech ASR Corpus
    Noise Source: Environmental Sound Classification 50
    
    File Statistics:
    ---------------
    Total Clean Speech Files: {len(clean_files)}
    Total Noise Files: {len(noise_files)}
    
    Audio Properties:
    ----------------
    Target Sample Rate: {TARGET_SR} Hz
    Target Duration: {DURATION} seconds
    Target Samples per Clip: {int(TARGET_SR * DURATION)}
    
    Clean Speech Duration:
      Mean: {np.mean(clean_durations):.2f}s ± {np.std(clean_durations):.2f}s
      Range: [{np.min(clean_durations):.2f}s, {np.max(clean_durations):.2f}s]
    
    Noise Duration:
      Mean: {np.mean(noise_durations):.2f}s ± {np.std(noise_durations):.2f}s
      Range: [{np.min(noise_durations):.2f}s, {np.max(noise_durations):.2f}s]
    
    Data Split (80/10/10):
    ---------------------
    Clean Speech:
      Train: {len(clean_train)} | Val: {len(clean_val)} | Test: {len(clean_test)}
    Noise:
      Train: {len(noise_train)} | Val: {len(noise_val)} | Test: {len(noise_test)}
    
    """)
    
    print("="*60)
    print("EDA Complete. All visualizations saved to src/Deliverable3/eda_images/")
    print("="*60)

def plot_amplitude_comparison(clean_files, noise_files, num_samples=100, save_dir="src/Deliverable3/eda_images"):    
    # Random samples
    clean_sample = random.sample(clean_files, min(num_samples, len(clean_files)))
    noise_sample = random.sample(noise_files, min(num_samples, len(noise_files)))
    
    clean_audios = []
    clean_stds = []
    clean_maxs = []
    clean_mins = []
    
    print(f"\nLoading {len(clean_sample)} clean audio files")
    for i, file in enumerate(clean_sample):
        try:
            audio = load_and_format(file)
            clean_audios.append(audio)
            clean_stds.append(np.std(audio))
            clean_maxs.append(np.max(audio))
            clean_mins.append(np.min(audio))
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(clean_sample)}...")
        except Exception as e:
            print(f"  Error: {e}")
    
    noise_audios = []
    noise_stds = []
    noise_maxs = []
    noise_mins = []
    
    print(f"\nLoading {len(noise_sample)} noisy audio files...")
    for i, file in enumerate(noise_sample):
        try:
            audio = load_and_format(file)
            noise_audios.append(audio)
            noise_stds.append(np.std(audio))
            noise_maxs.append(np.max(audio))
            noise_mins.append(np.min(audio))
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(noise_sample)}...")
        except Exception as e:
            print(f"  Error: {e}")
    

    fig = plt.figure(figsize=(14, 10))
    
    # 1. Boxplot standard deviation
    ax1 = plt.subplot(2, 2, 1)
    data_to_plot_std = [clean_stds, noise_stds]
    bp1 = ax1.boxplot(data_to_plot_std, labels=['Clean Speech', 'Noise'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightblue')
    bp1['boxes'][1].set_facecolor('lightcoral')
    ax1.set_ylabel('Standard Deviation')
    ax1.set_title('Amplitude Variation (Std Dev)')
    ax1.grid(True, alpha=0.3)
    
    # add media values
    ax1.text(1, np.mean(clean_stds), f'μ={np.mean(clean_stds):.4f}', 
             ha='center', va='bottom', fontsize=9)
    ax1.text(2, np.mean(noise_stds), f'μ={np.mean(noise_stds):.4f}', 
             ha='center', va='bottom', fontsize=9)
    
    # 2. Violin plot of amplitudes (distribution)
    ax2 = plt.subplot(2, 2, 2)
    sample_size = min(10, len(clean_audios), len(noise_audios))
    clean_sample_plot = random.sample(clean_audios, sample_size)
    noise_sample_plot = random.sample(noise_audios, sample_size)
    
    # forr each audio we take points for visualization
    clean_points = []
    for audio in clean_sample_plot:
        step = max(1, len(audio) // 1000)  # 1000 points per audio
        clean_points.extend(audio[::step][:1000])
    
    noise_points = []
    for audio in noise_sample_plot:
        step = max(1, len(audio) // 1000)
        noise_points.extend(audio[::step][:1000])
    

    parts = ax2.violinplot([clean_points, noise_points], positions=[1, 2], 
                           showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('lightblue')
    parts['bodies'][1].set_facecolor('lightcoral')
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Clean Speech', 'Noise'])
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'Amplitude Distribution (Sample of {sample_size} files)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 3. Comparative waveforms 
    ax3 = plt.subplot(2, 1, 2)
    
    # Select a representative audio (close to the median of std dev)
    clean_idx = np.argmin(np.abs(np.array(clean_stds) - np.median(clean_stds)))
    noise_idx = np.argmin(np.abs(np.array(noise_stds) - np.median(noise_stds)))
    
    clean_representative = clean_audios[clean_idx]
    noise_representative = noise_audios[noise_idx]
    
    # only first 2s or until 32000 samples
    max_samples = min(32000, len(clean_representative), len(noise_representative))
    time = np.arange(max_samples) / TARGET_SR
    
    ax3.plot(time, clean_representative[:max_samples], label='Clean Speech', alpha=0.7, color='blue', linewidth=0.8)
    ax3.plot(time, noise_representative[:max_samples], label='Noise', alpha=0.7, color='orange', linewidth=0.8)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Waveform Comparison (Representative Samples)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, min(2.0, max_samples / TARGET_SR)])
    
    plt.suptitle('Audio Analysis: Clean Speech vs Noise', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'amplitude_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    

    print(f"\n{'='*60}")
    print("AMPLITUDE STATISTICS SUMMARY")
    print(f"{'='*60}")
    print(f"\nClean Speech Statistics:")
    print(f"  Mean Std Dev:     {np.mean(clean_stds):.6f} ± {np.std(clean_stds):.6f}")
    print(f"  Amplitude Range:  [{np.min(clean_mins):.4f}, {np.max(clean_maxs):.4f}]")
    print(f"  Std Dev Range:    [{np.min(clean_stds):.6f}, {np.max(clean_stds):.6f}]")
    
    print(f"\nNoise Statistics:")
    print(f"  Mean Std Dev:     {np.mean(noise_stds):.6f} ± {np.std(noise_stds):.6f}")
    print(f"  Amplitude Range:  [{np.min(noise_mins):.4f}, {np.max(noise_maxs):.4f}]")
    print(f"  Std Dev Range:    [{np.min(noise_stds):.6f}, {np.max(noise_stds):.6f}]")
    
    print(f"\nCOMPARISON:")
    if np.mean(clean_stds) > np.mean(noise_stds):
        print(f"  Clean speech has {np.mean(clean_stds)/np.mean(noise_stds):.2f}x more amplitude variation than noise")
    else:
        print(f"  Noise has {np.mean(noise_stds)/np.mean(clean_stds):.2f}x more amplitude variation than clean speech")
    
    print(f"\nPlot saved to: {save_dir}/amplitude_comparison.png")
    print(f"{'='*60}")
    
    return clean_stds, noise_stds

def main():
    """Main EDA execution."""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("Audio Noise Suppression with Transformers")
    print("="*60)
    
    # download datasets
    clean_dir, noise_dir = download_datasets()
    
    # analyze file distribution
    clean_files, noise_files = analyze_file_distribution(clean_dir, noise_dir)
    
    # get the splits (usando directorios, no files)
    clean_train, clean_val, clean_test, noise_train, noise_val, noise_test = analyze_train_val_test_split(clean_dir, noise_dir)
    
    # analyze audio durations
    clean_durations, noise_durations = analyze_audio_durations(clean_files, noise_files)
    
    # plot duration distributions
    plot_duration_distributions(clean_durations, noise_durations)
    
    # plot split distribution
    plot_split_distribution(clean_train, clean_val, clean_test,
                           noise_train, noise_val, noise_test)
    
    # analyze SNR effect
    analyze_snr_effect(clean_files, noise_files)

    # compare amplitude distributions between clean and noise
    plot_amplitude_comparison(clean_files, noise_files)

    # calculate and display mean amplitude statistics
    print_audio_means(clean_files, noise_files)
    
    # generate summary report
    generate_summary_report(clean_dir, noise_dir, clean_files, noise_files,
                           clean_durations, noise_durations,
                           clean_train, clean_val, clean_test,
                           noise_train, noise_val, noise_test)

if __name__ == "__main__":
    main()