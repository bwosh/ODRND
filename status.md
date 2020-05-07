# Status

The project is in very early stage.  
Some concepts are being implemented.

Planned:
- [x] Coco dataset subset extraction
- [x] MobileNet v2 backbone model
- [ ] MobileNet v2 in SSDLite model
- [x] Anchors specification
- [ ] Training SSD with anchors
- [ ] TFLite export
- [ ] Quantization-aware training & TFLite quantized model
- [ ] Tests on ARM devices (CPU, NNAPI / FP32 FP16 INT8 ) 
- [ ] Tweaking the hyperparameters to find optimal mAP vs FPS tradeoff

Backlog:
- [ ] Training with heatmaps&sizes (CenterNet)
- [ ] More models to test & compare