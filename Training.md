# How the "Hey Tara" wake-word models were built

## Where the models come from

Both models are taken from published research on **small-footprint keyword spotting (SF-KWS)**: wake-word detectors small enough to run all the time on edge hardware. The main reference is:

> S. Garai and S. Samui, *"Advances in small-footprint keyword spotting for TinyML: A comprehensive review of efficient models and algorithms,"* Neurocomputing, vol. 695, 2026, art. 134028. https://www.sciencedirect.com/science/article/abs/pii/S0925231226014268

The review groups SF-KWS work into seven categories: model architecture, learning techniques, model compression, attention-aware architectures, feature optimization, neural architecture search, and hybrid approaches. It also explains why wake-word detection belongs on the device. Running the trigger word on a server would mean streaming speech to the cloud all the time, which harms privacy and uses more power. So a low-power keyword spotter has to run locally, using only the device's own compute and storage.

Following that framing, two compact convolutional architectures from the literature were chosen:

- **TC-ResNet** (Choi et al., Interspeech 2019): a residual network that uses temporal convolution only.
- **MatchboxNet** (Majumdar and Ginsburg, NVIDIA, Interspeech 2020): a network built from 1-D time-channel separable convolutions, derived from QuartzNet.

Both were first designed and benchmarked on Google Speech Commands, which is a multi-class task. Here they were re-implemented in PyTorch, changed to a single-logit binary detector ("hey tara" vs. everything else), and trained from scratch. No pretrained weights were used.

## Process overview

Training has three stages:

1. **Feature preparation.** Audio is cleaned, normalised, augmented and turned into log-mel spectrograms.
2. **Pre-training on synthetic speech.** The model learns from a large text-to-speech (TTS) corpus with many voices and engines.
3. **Fine-tuning on real recordings.** The model is adapted to real microphone audio, closing the gap between synthetic and real speech.

## Front end

- 16 kHz audio is converted to **40-band log-mel features** (25 ms windows, 10 ms hop) over a fixed **2-second window**. That gives a 40 × 201 time-frequency matrix, standardised per mel band.
- Silence is trimmed with an energy-based voice-activity detector. Loudness is normalised to a fixed RMS level, which removes about 12 dB of loudness difference between TTS engines that would otherwise act as an engine fingerprint.
- The word is **right-aligned** in the window over a low noise floor instead of digital silence. This matches what a streaming detector sees: the word has just ended.
- The negative examples include deliberately confusable speech: near-miss names, phrases that start like the wake phrase, partial phrases, and vowel-swapped variants.

## Architectures

Both networks follow the same idea. They treat the frequency axis as **channels** right away and do all further processing with cheap **1-D convolutions over time**. Each ends with global average pooling and a linear layer that outputs one logit.

### TC-ResNet (~490k parameters)

- A 1-D convolutional stem maps the 40 mel channels to 48.
- Three residual blocks widen the channels from 48 to 64, 96 and 128. Each block has 9-tap temporal convolutions with stride 2, BatchNorm, and a strided 1×1 skip connection.
- The time axis shrinks from 201 frames to about 26.
- Strength: much cheaper than 2-D CNNs that treat the spectrogram as an image, with a wide temporal receptive field.

### MatchboxNet-3×1×64 (~121k parameters)

- The prologue is a strided 1-D convolution from 40 to 128 channels.
- Three residual blocks (kernels 13, 15, 17) each use a **depthwise convolution over time followed by a pointwise convolution across channels**. This time-channel separable factorisation greatly reduces parameters and multiply-accumulates.
- The epilogue is a wide, dilated depthwise convolution (kernel 29, dilation 2) for long temporal context.
- Strength: the smallest model, with long context at very low parameter cost.

In short, TC-ResNet is simple, has more capacity and downsamples progressively. MatchboxNet is about 4× smaller and downsamples once, keeping finer temporal resolution.

## Training recipe

- **Objective and optimisation:** binary cross-entropy, AdamW, a one-cycle learning-rate schedule (peak 3e-3 for TC-ResNet, 2e-3 for MatchboxNet), effective batch size 512, mixed precision. The checkpoint with the best validation ROC-AUC is kept.
- **Speaker shortcut:** in the synthetic training data, each voice says only the wake phrase or only other phrases. A network could therefore learn *who* is speaking instead of *what* is said. Three measures counter this:
  - validation uses held-out voices that speak both classes;
  - augmentation blurs speaker identity;
  - metrics are reported with voice-level bootstrap confidence intervals.
- **Augmentation, in two layers:**
  - offline: room reverberation and additive noise at a range of SNRs;
  - online, on the GPU: gain, time shift, SpecAugment masking, frequency warping, spectral tilt and random EQ. The last three change the apparent vocal tract and microphone.
- **Threshold:** the decision threshold is not fixed at 0.5. It is set from negative examples so that it meets a target false-accept rate, which is how deployed detectors are tuned.

## Fine-tuning on real audio

- Each real clip is placed on a fixed 3-second canvas and split into overlapping 2-second windows. This stops clip length from giving away the label.
- **Multiple-instance learning:** the clip's score is the maximum over its windows, so the model learns both whether the word is present and where it is.
- **Synthetic replay** is mixed into every batch to prevent catastrophic forgetting. The backbone uses a low learning rate and the output head a slightly higher one.
- Evaluation uses **session-grouped cross-validation**, and the threshold is calibrated on real negatives with sliding-window scoring, as in live use.

## Evaluation

The main metric is **FRR at a fixed FAR**: how often the wake word is missed for a given false-alarm budget. EER and ROC-AUC are also reported, along with model cost (parameters, MACs, CPU latency), since the target is an edge device.
