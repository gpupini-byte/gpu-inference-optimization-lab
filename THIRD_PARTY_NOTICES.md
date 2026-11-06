# Third-party models, software, and image attribution

## TorchVision ResNet-18 pretrained weights

The benchmark downloads `torchvision.models.ResNet18_Weights.DEFAULT` at runtime; model weights are not included in this repository. TorchVision notes that pretrained models may have license or terms inherited from their training datasets, and users are responsible for checking permission for their use case. See the [TorchVision pretrained models documentation](https://docs.pytorch.org/vision/stable/models.html). These ImageNet-trained weights and ImageNet class labels are separate third-party materials and are not relicensed by this repository's MIT license. The benchmark's ImageNet classes and weights are used for local inference experiments.

## Validation images

The three redistributed demo images are recorded with source, author, license, and SHA-256 hash in [`data/real_validation/manifest.json`](data/real_validation/manifest.json). They were obtained from Wikimedia Commons and re-encoded for the validation set:

- `labrador_dog.jpg`: Karen Arnold, CC0 1.0, [source page](https://commons.wikimedia.org/wiki/File:Labrador_dog.jpg).
- `tabby_cat.jpg`: Ubuntuist, released into the public domain by the copyright holder, [source page](https://commons.wikimedia.org/wiki/File:Tabby-cat.jpg).
- `goldfish.jpg`: Tokumeigakarinoaoshima, CC0 1.0, [source page](https://commons.wikimedia.org/wiki/File:Goldfish,_which_has_been_brought_up_for_about_15_years.JPG).

The image set's purpose and limitations are described in [`data/real_validation/README.md`](data/real_validation/README.md). The original licenses and source-page terms apply to those images.

## Dependencies

PyTorch, TorchVision, NumPy, Pillow, Matplotlib, pytest, ONNX, ONNX Script, and ONNX Runtime GPU are separately licensed projects. Their licenses remain with their respective authors and distributors; installing dependencies does not transfer them under this repository's license.
