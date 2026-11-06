# Real image validation subset

This tiny set is only a smoke check for image preprocessing and output consistency. It is too small to support a general classification-accuracy claim. Each expected label is an ImageNet-1K class chosen from the Wikimedia Commons description. The runner reports exact-label top-1 accuracy separately from prediction agreement with the PyTorch FP32 output.

All selected files are public domain or CC0 according to their Commons pages. Provenance is recorded in `manifest.json`; the originals are hosted by Wikimedia Commons:

- Labrador dog by Karen Arnold, CC0 1.0: <https://commons.wikimedia.org/wiki/File:Labrador_dog.jpg>
- Tabby cat by Ubuntuist, released into the public domain: <https://commons.wikimedia.org/wiki/File:Tabby-cat.jpg>
- Common goldfish by Tokumeigakarinoaoshima, CC0 1.0: <https://commons.wikimedia.org/wiki/File:Goldfish,_which_has_been_brought_up_for_about_15_years.JPG>

The benchmark uses synthetic deterministic tensors. This validation set runs separately and does not affect performance measurements.
