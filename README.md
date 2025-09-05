# IaI-SimCLR

Implementation of [Multi-Modal Multi-Objective Contrastive Learning for Sentinel-12 Imagery (CVPRW 2023)](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Prexl_Multi-Modal_Multi-Objective_Contrastive_Learning_for_Sentinel-12_Imagery_CVPRW_2023_paper.pdf)

# Usage

To run the script with the three configuration files, use the following commands:

```bash
python main.py --config configs/DualSimCLR.yaml
python main.py --config configs/IaI_SimCLR_noColorAugmentation.yaml
python main.py --config configs/IaI_SimCLR.yaml
```

# Citation

If you use this work, please cite:

```bibtex
@inproceedings{prexl2023multi,
  title={Multi-modal multi-objective contrastive learning for {Sentinel-1/2} imagery},
  author={Prexl, Jonathan and Schmitt, Michael},
  booktitle={Proceedings of CVPR Workshops},
  pages={2135--2143},
  year={2023},
}
```

# License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.