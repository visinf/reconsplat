# Variational Gaussian Rasterizer

Rasterization kernel used in **ReconSplat: Generalizable 3D Scene Reconstruction Beyond Observed Views**.

This implementation is adapted from the 3D Gaussian rasterizer used in <a href="https://vip.mpi-inf.mpg.de/latentsplat/">**latentSplat: Autoencoding Variational Gaussians for Fast Generalizable 3D Reconstruction**</a> (Wewer et al.), which is itself based on the original differentiable rasterizer introduced in **3D Gaussian Splatting for Real-Time Rendering of Radiance Fields** (Kerbl et al.).

## Installation

Run the following commands to build it from source:
```bash
conda config --set channel_priority flexible
conda install -c 'nvidia/label/cuda-11.8.0' cuda-toolkit=11.8.0
pip install --no-build-isolation .
````

## Citation

If you can make use of this code in your own work, please be so kind to cite our paper and other works this code has been derived from. 

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre>
      <code>
@inproceedings{stracquadanio2026reconsplat,
    title   = {{ReconSplat}: {G}eneralizable {3D} scene reconstruction beyond observed views},
    author  = {Stracquadanio, Giuseppe and Raj, Kevin and Grabinski, Julia and Roth, Stefan},
    booktitle = {{ECCV}},
    year    = {2026},
}
      </code>
    </pre>
    <pre>
      <code>
@inproceedings{wewer2024latentsplat,
  title={{latentSplat}: {A}utoencoding variational {Gaussians} for fast generalizable {3D} reconstruction},
  author={Wewer, Christopher and Raj, Kevin and Ilg, Eddy and Schiele, Bernt and Lenssen, Jan Eric},
  booktitle={{ECCV}},
  volume={15145},
  pages={456--473},
  year={2024},
  doi = {10.1007/978-3-031-73021-4_27}
}
      </code>
    </pre>
    <pre>
      <code>
@article{kerbl20233d, 
    title = {{3D} {Gaussian} splatting for Real-Time Radiance Field Rendering}, 
    author = {Kerbl, Bernhard and Kopanas, Georgios and Leimkuehler, Thomas and Drettakis, George}, 
    year = {2023},
    volume = {42},
    number = {4},
    journal = {ACM Trans. Graph.}, 
    numpages = {14},
    doi = {10.1145/3592433}
}
      </code>
    </pre>
  </div>
</section>
