# Target devices 
Performance tests is planned to be run one the low-end devices listed here:

- Rasprebby Pi Zero W (Broadcom BCM2835, ARMv6Z (32-bit), **1× ARM1176JZF-S** 1 GHz)
- Xiaomi Redmi 7A (Snapdragon 439, ARMv8-A (64/32-bit), **8x Cortex-A53** 2 GHz + Adreno 505 + Hexagon 536)

# Tweaks to check:
- Raspberry Pi VideoCore IV GPU usage on Rasprebby Pi Zero W
- Android: NNAPI detelate TFLite, GPU delegate TFLite, Hexagon delegate TFLite

# Further comparisons
Depending on time consumption of development process, more tests will be conducted on:

- Rasprebby Pi 2 (Broadcom BCM2836, ARMv7-A (32-bit), **4× Cortex-A7** 900 MHz)
- Raspberry Pi 3B+ (Broadcom BCM2837B0, ARMv8-A (64/32-bit), **4× Cortex-A53** 1.4 GHz)
- Raspberry Pi 4 (Broadcom BCM2711, ARMv8-A (64/32-bit), **4× Cortex-A72** 1.5 GHz)
- Google Coral Dev Board (NXP i.MX 8M SoC, ARMv8-A (64/32-bit), **4x Cortex-A53** 1.5 GHz +  Cortex-M4F + Edge TPU)
- NVidia Jetson Nano (**4x Cortex-A57** + NVIDIA Maxwell GPU)

including:
- EDGE TPU usage
- Tensorflow on GPU/CUDA (Jetson) 
- TensorRT usage (Jetson)