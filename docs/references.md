# References and library acknowledgements

Agentic Active AutoResearch implements a practical engineering combination; it does not claim authorship of the
underlying active-learning, molecular representation, uncertainty, or agent-tool ideas.

## Active learning and Bayesian optimization

1. Burr Settles. *Active Learning Literature Survey*. University of Wisconsin–Madison Computer
   Sciences Technical Report 1648, 2009. [PDF](https://burrsettles.com/pub/settles.activelearning_20090109.pdf)
2. Maximilian Balandat et al. *BoTorch: A Framework for Efficient Monte-Carlo Bayesian
   Optimization*. NeurIPS 2020. [Paper](https://proceedings.neurips.cc/paper/2020/hash/f5b1b89d98b7286673128a5fb112cb9a-Abstract.html)
3. David E. Graff et al. *Accelerating high-throughput virtual screening through molecular
   pool-based active learning*. Chemical Science 12 (2021).
   [DOI](https://doi.org/10.1039/D0SC06805E)
4. James Bergstra and Yoshua Bengio. *Random Search for Hyper-Parameter Optimization*. JMLR 13,
   281–305 (2012). [JMLR](https://www.jmlr.org/papers/v13/bergstra12a.html)
5. Gavin C. Cawley and Nicola L. C. Talbot. *On Over-fitting in Model Selection and Subsequent
   Selection Bias in Performance Evaluation*. JMLR 11, 2079–2107 (2010).
   [JMLR](https://www.jmlr.org/papers/v11/cawley10a.html)

Settles provides the pool-based AL vocabulary used here. BoTorch is a recommended extension for
fully Bayesian and multi-objective acquisition. MolPAL is the closest molecular pool-screening
reference; Agentic Active AutoResearch adds provider-neutral policy guardrails and a generic task/oracle boundary.
Bergstra–Bengio motivates random search as the minimum matched-budget comparison for an adaptive
training-candidate generator. Cawley–Talbot motivates the strict separation between labeled-only
inner model selection and outer performance reporting; it also cautions that the selection
criterion itself can be overfit.

## Training curricula and robust objectives

6. Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. *Curriculum Learning*.
   ICML, 41–48 (2009). [DOI](https://doi.org/10.1145/1553374.1553380)
7. Peter J. Huber. *Robust Estimation of a Location Parameter*. The Annals of Mathematical
   Statistics 35, 73–101 (1964). [DOI](https://doi.org/10.1214/aoms/1177703732)

Bengio et al. motivate easy-to-hard presentation; this package implements a deterministic,
training-only centroid-distance curriculum with a matched optimizer-step budget rather than
claiming to reproduce a particular paper experiment. Huber motivates the robust objective; the
code uses PyTorch's [`SmoothL1Loss(beta=1.0)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html),
which PyTorch documents as closely related to Huber loss.

## Molecular prediction and uncertainty

8. Kevin Yang et al. *Analyzing Learned Molecular Representations for Property Prediction*.
   JCIM 59(8), 2019. [DOI](https://doi.org/10.1021/acs.jcim.9b00237)
9. Esther Heid et al. *Chemprop: A Machine Learning Package for Chemical Property Prediction*.
   JCIM 64(1), 9–17 (2024). [DOI](https://doi.org/10.1021/acs.jcim.3c01250)
10. David E. Graff et al. *Chemprop v2: An Efficient, Modular Machine Learning Package for
   Chemical Property Prediction*. JCIM 66(1), 28–33 (2026).
   [DOI](https://doi.org/10.1021/acs.jcim.5c02332)
11. Alexander Amini et al. *Evidential Deep Learning for Guided Molecular Property Prediction and
   Discovery*. ACS Central Science, 2021. [DOI](https://doi.org/10.1021/acscentsci.1c00546)
12. Yarin Gal and Zoubin Ghahramani. *Dropout as a Bayesian Approximation: Representing Model
   Uncertainty in Deep Learning*. ICML 2016.
   [PMLR](https://proceedings.mlr.press/v48/gal16.html)
13. Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. *Simple and Scalable
   Predictive Uncertainty Estimation using Deep Ensembles*. NeurIPS 2017.
   [Paper](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)
14. David Rogers and Mathew Hahn. *Extended-Connectivity Fingerprints*. JCIM 50(5), 2010.
    [DOI](https://doi.org/10.1021/ci100050t)
15. Guy W. Bemis and Mark A. Murcko. *The Properties of Known Drugs. 1. Molecular Frameworks*.
    Journal of Medicinal Chemistry 39(15), 1996.
    [DOI](https://doi.org/10.1021/jm9602928)

The Chemprop route is optional. The built-in MLP uses ensemble + MC Dropout uncertainty and does
not claim calibrated uncertainty or equivalence to evidential uncertainty. Morgan fingerprints
and scaffold grouping follow the Rogers–Hahn and Bemis–Murcko lineages through RDKit.

## Agentic science context

16. Andres M. Bran et al. *Augmenting large language models with chemistry tools*. Nature Machine
   Intelligence 6, 525–535 (2024). [DOI](https://doi.org/10.1038/s42256-024-00832-8)
17. Chris Lu et al. *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery*.
   2024. [arXiv](https://arxiv.org/abs/2408.06292)
18. Samuel Schmidgall et al. *Agent Laboratory: Using LLM Agents as Research Assistants*. 2025.
    [arXiv](https://arxiv.org/abs/2501.04227)

These works motivate tool-grounded and human-audited research agents. Agentic Active AutoResearch intentionally
targets the narrower, measurable decision of which pool items to label next.

## Libraries

- [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), and
  [scikit-learn](https://scikit-learn.org/) provide numerical, table, preprocessing, metrics, split,
  and baseline components.
- [PyTorch](https://pytorch.org/) provides CUDA/MPS training and MC Dropout execution.
- [NumPy NPZ I/O](https://numpy.org/doc/stable/user/how-to-io.html) provides the built-in
  checkpoint container; NumPy recommends `allow_pickle=False` for security and portability.
- [PyTorch state dictionaries](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
  define the weight-name contract reconstructed from verified arrays; the built-in continuation
  path does not call `torch.load`.
- [Pydantic](https://docs.pydantic.dev/) validates configuration contracts.
- [PyYAML](https://pyyaml.org/) parses YAML.
- [RDKit](https://www.rdkit.org/) provides optional Morgan fingerprints and molecular utilities.
- [Chemprop](https://github.com/chemprop/chemprop) provides the optional D-MPNN backend.
- [Apache Arrow/PyArrow](https://arrow.apache.org/docs/python/) provides optional Parquet I/O.

Consult each dependency and dataset for its current license and citation instructions.
