import audioread
import numpy as np
import pyaudio
import matplotlib.pyplot as plt


def playAudio(PCM, meta_data):
    """
    play the audio from the PCM
    :param PCM:
    :param meta_data:
    """
    p = pyaudio.PyAudio()
    # open stream (2), 2 is size in bytes of int16
    stream = p.open(format=p.get_format_from_width(2),
                    channels=meta_data["channels"],
                    rate=meta_data["samplerate"],
                    output=True)
    
    # play stream (3), blocking call
    stream.write(PCM)
    
    # stop stream (4)
    stream.stop_stream()
    stream.close()
    
    # close PyAudio (5)
    p.terminate()


metaData = {}
c = 0
with audioread.audio_open("audio2.m4a") as f:
    print("f.channels: ", f.channels, ", f.samplerate: ", f.samplerate, ", f.duration: ", f.duration)
    
    metaData["channels"] = f.channels
    metaData["samplerate"] = f.samplerate
    metaData["duration"] = f.duration
    
    PCM = np.ndarray(shape=(int(f.samplerate * f.duration), ), dtype=np.int16)
    for buf in f:
        #print(len(buf))
        for a in range(0, len(buf) - 1, 2):
            # print((buf[a + 1] << 8) | buf[a])
            if (c < f.samplerate * f.duration):
                PCM[c] = (buf[a + 1] << 8) | buf[a]
            c += 1
    # print('\n', c)
    print(PCM)

#playAudio(PCM, metaData)




NFFT = 128
overlap = 0

N_audio_samples = int(metaData["duration"] * metaData["samplerate"])
st = 1.0 / metaData["samplerate"]  # Sample time
bins = np.fft.fftfreq(NFFT, st)[:int(NFFT/2)]
print(bins)

img_shape = (bins.shape[0], int((metaData["duration"]*metaData["samplerate"])/NFFT))

real = np.ndarray(shape=img_shape, dtype=np.int16)

imag = np.ndarray(shape=img_shape, dtype=np.int16)

magnitude = np.ndarray(shape=img_shape, dtype=np.int16)

for i in range(img_shape[0]):
    # FFT + bins + normalization
    fft = np.array([i / (NFFT / 2) for i in np.fft.fft(PCM[i:i+NFFT-overlap])])
    
    print(fft.shape)

    real[:, i] = np.real(fft[:int(NFFT / 2)])
    
    #axes[1][n_plot].plot(bins[:int(step / 2)], np.real(fft[:int(step / 2)]), 'b-')

    imag[:, i] = np.imag(fft[:int(NFFT / 2)])
    #axes[2][n_plot].plot(bins[:int(step / 2)], np.imag(fft[:int(step / 2)]), 'b-')

    magnitude[:, i] = np.abs(fft[:int(NFFT / 2)])
    #axes[3][n_plot].plot(bins[:int(step / 2)], np.abs(fft[:int(step / 2)]), 'b-')

plt.figure(1)
plt.axis((0, metaData["duration"], 0, 8000))
plt.axes(aspect=metaData["duration"]/8000)
plt.imshow(real)
print(real.shape)

plt.figure(2)
Pxx, freqs, bins, im = plt.specgram(PCM, Fs=metaData["samplerate"], NFFT=128, noverlap=0, mode= 'psd')
print(Pxx.shape)
print(bins.shape)
print(freqs.shape)
plt.show()
