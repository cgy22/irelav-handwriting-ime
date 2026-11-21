# Third-Party Notices

This project would not exist without the following work. Licenses are reproduced
or summarized below; where a dataset ships no formal license, the upstream terms
are stated as published.

---

## Datasets (used for training, not redistributed)

### MNIST
* **Authors**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
* **Source**: <http://yann.lecun.com/exdb/mnist/>
* **Terms**: MNIST is published for research and educational use. The upstream
  page states no formal open-source license. It is *not* redistributed here —
  only statistical weights derived from training are shipped.
* **Please cite**: Y. LeCun, L. Bottou, Y. Bengio, P. Haffner,
  "Gradient-based learning applied to document recognition",
  *Proceedings of the IEEE*, 86(11):2278–2324, 1998.

### EMNIST
* **Authors**: Gregory Cohen, Saeed Afshar, Jonathan Tapson, André van Schaik
* **Source**: NIST Special Database 19, packaged as EMNIST and distributed by
  NIST (<https://www.nist.gov/itl/products-and-services/emnist-dataset>).
* **Terms**: released by NIST for research use. Not redistributed here.
* **Please cite**: G. Cohen, S. Afshar, J. Tapson, A. van Schaik,
  "EMNIST: an extension of MNIST to handwritten letters", arXiv:1702.05373, 2017.

### Font-derived synthetic glyphs
The synthetic training glyphs are *rendered at build time* from fonts installed
on the local machine (Arial, Times New Roman, Calibri, Segoe UI, Georgia,
Verdana, Trebuchet MS, Courier New — all Microsoft Corporation). No font files
are redistributed. The rendered bitmaps are used only as training data inside
this project.

---

## Software

### WebView2 SDK / WebView2 Runtime — Microsoft Corporation
`Microsoft.Web.WebView2.Core.dll`, `Microsoft.Web.WebView2.WinForms.dll` and
`WebView2Loader.dll` are redistributed as part of the WebView2 SDK under the
Microsoft license that accompanies the SDK package
(`.vendor/webview2/pkg/LICENSE.txt`). The WebView2 *runtime* is a component of
Microsoft Edge and is installed by Windows; it is not bundled here.

### PyTorch — Meta Platforms, Inc.
BSD-3-Clause. <https://github.com/pytorch/pytorch/blob/main/LICENSE>

### NumPy — NumPy Developers
BSD-3-Clause. <https://numpy.org/doc/stable/license.html>

### Pillow (PIL) — Jeffrey A. Clark and contributors
HPND / PIL license. <https://python-pillow.org/>

### SciPy — SciPy Developers
BSD-3-Clause. <https://scipy.org/>

### Matplotlib — Matplotlib Development Team
PSF-based license, BSD-compatible. <https://matplotlib.org/stable/users/project/license.html>

### fontTools — Just van Rossum, Behdad Esfahbod and contributors
MIT. <https://github.com/fonttools/fonttools/blob/main/LICENSE>

### .NET Framework / Windows Forms — Microsoft Corporation
Part of Windows; used for the desktop shell.

---

## Assets

### TonhTze AI logos
`assets/tonhtze-ai-light.svg` and `assets/tonhtze-ai-dark.svg` are derived from
the TonhTze AI logo set. The original files used a `<text>` element that
depended on the Cascadia Code font; the copies here replace that text with
vector paths traced from **Consolas Bold** so the mark renders identically on
any machine without font dependencies.

### Word list
`data/words.json` is generated from `data/source/wordData.js`, the constructed
language dictionary provided by Dorn Hub. It is distributed under the same MIT
license as this project.

---

## What this project does *not* bundle

* MNIST / EMNIST raw data (downloaded separately by the user)
* Font files
* The .NET SDK, Visual Studio, or the Windows SDK
* Any Edge browser binaries
