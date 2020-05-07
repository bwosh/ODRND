# Object Detection Research & Development

This open-source repository is a space for Object Detecton Research with ARM-based devices performance in mind.

![object detection sample](./assets/sample_bbox.jpg)

# Goals
1. Push low-end Android device (**[Xiaomi Redmi 7A](https://www.gsmarena.com/xiaomi_redmi_7a-9716.php)**) to its object detection (3 classes) FPS limits using:
    - [recent findings in Computer Vision](./assets/references.md#readme)
    - TFLite delegates for: [NNAPI](https://www.tensorflow.org/lite/performance/nnapi), [GPU](https://www.tensorflow.org/lite/performance/gpu), [Hexagon](https://www.tensorflow.org/lite/performance/hexagon_delegate)
2. Push **[Raspberry Pi Zero W](https://www.raspberrypi.org/products/raspberry-pi-zero-w/)** to its object detection (3 classes) FPS limits using:
    - [recent findings in Computer Vision](./assets/references.md#readme)
    - OpenCL implementation on [VideoCore IV GPU](https://github.com/doe300/VC4CL)

*3 classes mentioned above are: person, vehicle, animal*
# Details

You can navigate to the details here:

- **STATUS**: [Current status of the project](./assets/status.md#readme)
- **SRC**: [Training the networks/Using the code](./src#readme)
- **PERFORMANCE**: [Target devices and performance tests](./performance#readme)
- **REFERENCES**: [List of inspiring papers](./assets/references.md#readme)

# License

**ODRND - Object Detection Research And Development**  
Copyright (C) 2020  [Błażej Matuszewski](https://github.com/bwosh)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

If you need to use software in proprietary, non-open-source code: [licence exceptions](https://www.fsf.org/blogs/rms/selling-exceptions) may be obtained after contacting me.