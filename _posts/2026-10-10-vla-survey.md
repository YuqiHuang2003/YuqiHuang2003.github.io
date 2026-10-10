---
title: "Vision–Language–Action Models for Embodied Control: Integration, Applications, and Evaluation"
date: 2026-10-10
permalink: /blog/vla-survey/
excerpt: "A review of four VLA integration patterns, with a focus on action representations, feedback, correction, and the limits of reported robot-control results."
classes: wide vla-survey
categories:
  - Research
tags:
  - Vision-Language-Action
  - Robotics
  - Embodied AI
comments: false
share: false
related: false
---

## Abstract

This review compares four integration patterns for vision–language–action (VLA) models: modular planning, autoregressive action prediction, generative action chunks, and intermediate reasoning. The discussion focuses on robot manipulation, with reinforcement learning and hierarchical control as extensions and navigation and virtual-world agents as additional applications. FAST reduces training cost but takes longer to decode an action chunk than π₀. Reliable execution also requires timely feedback and opportunities to correct errors: intermediate predictions let operators revise a target or motion, and hierarchical controllers incorporate fresh images between slower reasoning steps. Long-horizon performance remains a challenge, particularly in preserving task progress and revising predicted actions when execution deviates from the plan.

<details class="vla-toc" markdown="1">
<summary>Contents</summary>

* Contents
{:toc}

</details>

## Introduction

### Concepts and Development

Vision–language–action models extend visual and language understanding to control. A vision–language model (VLM) can identify the cup named in an instruction, but converting that understanding into a successful lift also requires a gripper policy. RT-1 already uses language instructions to guide movement [[1]](#ref-rt1), unlike visuomotor policies such as Diffusion Policy [[2]](#ref-diffpolicy). Modern VLAs adapt representations learned from large-scale vision–language data [[3]](#ref-survey). Users can describe tasks in familiar language, and the robot can draw on knowledge beyond its demonstrations. This transfer may reduce the need to collect a separate dataset for every task.

CLIPort (CoRL 2021) combined pretrained semantic features with spatial manipulation features [[4]](#ref-cliport). PaLM-E (ICML 2023) integrated images and robot states into language-model reasoning [[5]](#ref-palme). RT-2 (CoRL 2023) then made robot actions part of a VLM's output vocabulary [[6]](#ref-rt2). Octo (RSS 2024) and OpenVLA (CoRL 2024) opened generalist policies to adaptation [[7]](#ref-octo), [[8]](#ref-openvla). Subsequent work explored richer action representations, including π₀'s flow-based chunks and FAST's compressed tokens (RSS 2025) [[9]](#ref-pi0), [[10]](#ref-fast). More recent research combines these policies with interaction-based learning and hierarchical control.

### Selection and Scope

This review groups eight core conference studies from 2023–2026 by their main contribution: modular planning (VoxPoser), autoregressive prediction (RT-2, OpenVLA, and FAST), generative chunks (π₀ and Discrete Diffusion VLA), and intermediate reasoning (RT-H and ECoT). The selection emphasizes contrasting action designs and experiments that test individual components. The discussion focuses on manipulation, in which RT-2 and OpenVLA adapted VLMs for robot control, and the decoder studies tested alternative action representations. Liu et al.'s NeurIPS study examines learning through interaction, and HiRT's CoRL experiments examine hierarchical control [[11]](#ref-rlvla), [[12]](#ref-hirt).

Two surveys summarize the development of VLA models and the difficulties of robot deployment [[3]](#ref-survey), [[13]](#ref-deploysurvey). Recent technical reports and company demonstrations supplement the conference studies. The application discussion also includes virtual-world agents.

### Main Findings

Faster action prediction helps a robot respond to a displaced object only if new observations can change the next movement. In π₀, a coordinated prefix executes from one observation before replanning [[9]](#ref-pi0). HiRT's smaller visual controller can adjust movement between full VLM passes by using a fresh image without recomputing the slower task features [[12]](#ref-hirt).

An intermediate prediction helps with correction by identifying what the operator needs to change. In ECoT, this can be an object's estimated location before the next reach [[14]](#ref-ecot). RT-H's language motions are easy to interpret, yet its action clusters work better on some precise movements [[15]](#ref-rth). A label can be readable without being the best choice for control. Liu et al.'s object-repositioning experiment tests a later stage of correction, after the reach has already started [[11]](#ref-rlvla).

## Detailed Discussion of Integration Approaches

The four groups describe where perception and instructions enter action prediction. Cross-modal pretraining is common to all four, and the groups overlap when a paper changes only part of a policy. FAST predicts chunks autoregressively; RT-H and ECoT add intermediate predictions to an autoregressive controller. RL can extend these policies through interaction. Hierarchical control allows task reasoning and motor prediction to run at different frequencies.

### Multimodal Understanding and Its Use in Control

Recognizing a cup does not tell a robot how firmly to grasp it. Pretraining helps interpret the object and instruction, but the robot's experience must connect that meaning to a feasible movement. RT-2 learns from both vision–language and robot tasks during joint fine-tuning [[6]](#ref-rt2). OpenVLA combines DINOv2 and SigLIP features within its language model context, enabling action prediction to leverage spatial and semantic cues [[8]](#ref-openvla). The policy can then reuse familiar visual and language patterns when adapting for new tasks.

The models turn this context into movement in different ways. Explicit spatial restrictions in VoxPoser can prevent a planner from taking a direct route through an obstacle [[16]](#ref-voxposer). End-to-end policies learn either a discrete control vocabulary or, in π₀, a continuous action expert conditioned on the scene and instruction [[9]](#ref-pi0). RT-H and ECoT predict a language motion or scene-specific information before low-level control [[15]](#ref-rth), [[14]](#ref-ecot). If the predicted motion or object location is wrong, an operator can correct it before the next reach.

During RL training, scene and language features help the policy select an action and estimate its return. Interaction then supplies feedback for policy updates [[11]](#ref-rlvla). At deployment, HiRT's smaller visual policy reads fresh images and cached task context, so it can adjust movement before the next VLM pass [[12]](#ref-hirt).

### Methods for Integrating Vision, Language, and Action

[Figure 1](#fig-interfaces) shows representative implementations of the four patterns and the two extensions. The diagrams separate their main computations, although a complete system may combine them.

<figure id="fig-interfaces" class="vla-figure">
  <a href="{{ '/images/blog/vla-integration-patterns.svg' | relative_url }}" aria-label="Open the full-size diagram">
    <img src="{{ '/images/blog/vla-integration-patterns.svg' | relative_url }}" alt="Four vision–language–action integration patterns: VoxPoser value maps and planning; RT-2 and OpenVLA autoregressive action tokens; π₀ flow-based action chunks; RT-H and ECoT intermediate reasoning with human correction. Two extensions show reinforcement learning and hierarchical control with fresh observations." width="1120" height="1250" loading="lazy">
  </a>
  <figcaption><strong>Figure 1.</strong> Four integration patterns and two extensions. Panels (a)–(d) show representative paths from observations to actions. In (c), π₀'s VLM and action expert interact through shared attention. The lower diagrams distinguish reward-based policy learning from hierarchical execution.</figcaption>
</figure>

#### Modular Perception and Planning

VoxPoser converts a command such as “open the drawer but keep away from the vase” into a path-planning problem with a no-go region. Language-generated code calls visual tools to construct 3D value maps, which a model-based planner uses to generate end-effector trajectories and replan as observations change [[16]](#ref-voxposer). The division between language-defined goals and externally planned movement makes this a modular form of VLA integration, without training an end-to-end action policy.

Recomputing the maps lets the planner follow a displaced target. In VoxPoser's disturbance experiment, success falls from 88% to 70%, compared with complete failure for the predefined-primitives baseline [[16]](#ref-voxposer). The planner can also use a local dynamics model updated online to plan object interactions.

#### Autoregressive Action Prediction

RT-2 quantizes controls into tokens and predicts them sequentially from the image, instruction, and generated prefix. Its two variants average 62% success, compared to RT-1's 32% in generalization evaluations [[6]](#ref-rt2). OpenVLA uses a related design in an open 7B-parameter model trained on approximately 970,000 demonstrations. Users can adapt it through low-rank fine-tuning [[8]](#ref-openvla).

FAST transforms correlated motions into cosine coefficients and applies quantization and byte-pair encoding, reducing a one-second table-bussing chunk from 140 to 28 tokens [[10]](#ref-fast). On the generalist data mixture, π₀-FAST reaches comparable performance to π₀ with five times fewer GPU hours. In the paper's NVIDIA RTX 4090 comparison, decoding a one-second chunk takes approximately 750 milliseconds with π₀-FAST and at most 100 milliseconds with π₀'s generative action expert [[10]](#ref-fast).

#### Generative Action Chunks

In π₀, an action expert receives robot state and noisy actions and attends to visual and language context. Flow updates transform the noise into a continuous action chunk, with a prefix executed before the next observation. On the reported 50 Hz robots, this amounts to 25 actions over 0.5 seconds [[9]](#ref-pi0). A target displacement during that interval, therefore, affects the next prediction rather than the prefix already being executed.

Discrete Diffusion VLA first resolves confident predictions as it refines noisy action tokens within one Transformer [[17]](#ref-discrete). Its principal decoder comparisons use LIBERO simulation, alongside small real-robot tests. With shared Qwen initialization and training, success is 81.2% for autoregression, 94.2% for parallel decoding, and 95.1% for discrete diffusion [[17]](#ref-discrete).

Parallel decoding is a reasonable starting point for these nominal tasks because it avoids repeated refinement and trails discrete diffusion by only 0.9 percentage points. In separate OpenVLA-based LIBERO-Goal tests, language changes reduce discrete diffusion's success by 0.8 percentage points, compared with 8.0 for parallel decoding. Under visual changes, the corresponding drops are 20.4 and 22.6 points, a much smaller gap; continuous diffusion loses 29.0 points [[17]](#ref-discrete).

#### Intermediate Reasoning and Correction

RT-H conditions low-level actions on a predicted language motion, such as moving forward. A shared VLM reads the image and original instruction at both stages [[15]](#ref-rth). Replacing motion descriptions with integer labels hurts performance. Action clusters perform slightly worse overall but do better on opening and closing a pistachio jar. The authors suggest that language labels are easier to predict across tasks, whereas action-derived clusters capture finer movement distinctions for the jar tasks [[15]](#ref-rth).

ECoT ties its reasoning supervision to scene-specific image and robot-state annotations. With the same VLM backbone and Bridge V2 robot data as its OpenVLA baseline, it gains approximately 28 percentage points in real WidowX trials [[14]](#ref-ecot). An operator can correct these predictions before an incorrect object location causes the arm to move toward the wrong place.

#### Extension I: Learning Objectives through RL

Liu et al. fine-tune OpenVLA with proximal policy optimization for instruction-conditioned pick-and-place in ManiSkill [[11]](#ref-rlvla). When the simulator teleports the object to a new table position at the fifth timestep, RL achieves 74.5% success, compared with 28.6% for supervised fine-tuning. The visualized rollouts show a supervised policy re-targeting the object but sweeping it off the table, and an RL policy completing placement [[11]](#ref-rlvla).

The reposition result is the clearest evidence of recovery in this pick-and-place setting because identifying the target again must be followed by a revised reach and successful placement.

The 2025 technical report on π<sup>*</sup><sub>0.6</sub> examines real-robot learning with RECAP, short for RL with Experience and Corrections via Advantage-conditioned Policies [[18]](#ref-recap). RECAP combines demonstrations with autonomous rollouts and expert interventions. A value model estimates task progress, and advantage-conditioned learning favors better actions. Laundry and espresso experiments measure successful work per hour and completion.

#### Extension II: Hierarchical Control

HiRT (CoRL 2024) uses slowly updated VLM features to guide a faster visual policy that reads fresh images. It nearly doubles the policy's inference frequency with comparable success in quasi-static tests. When targets move at 1 cm/s, success rises from Vanilla-VLA's 48% to HiRT's 75% [[12]](#ref-hirt). Fresh views may matter more on moving targets because a delayed observation can guide the arm toward an outdated position. In a visualized trial, HiRT catches a moving blue block that the slower baseline misses.

GR00T N1's technical preprint combines a pretrained VLM with a flow-based action model and embodiment-specific state and action mappings [[19]](#ref-gr00t). It trains on robot data, human video, and synthetic trajectories. Its adaptation results also expose a cost that new-task scores would miss. After post-training on right-hand-only demonstrations, the policy no longer performs a left-to-right handoff that the pretrained checkpoint could manage [[19]](#ref-gr00t).

Figure's 2026 Helix 02 company report adds a learned whole-body controller beneath task-level and motor policies [[20]](#ref-helix). The motor policy reads cameras, tactile sensors, and proprioception. A kitchen demonstration combines walking with loading the dishwasher.

### Applications and Evaluation

#### Robot Manipulation

Manipulation ranges from short tabletop actions to extended workflows such as laundry folding and box assembly [[9]](#ref-pi0). Value maps encode obstacle-avoidance constraints, and an operator can correct intermediate object predictions before they guide a reach [[16]](#ref-voxposer), [[14]](#ref-ecot). OpenVLA enables users to adapt instruction-conditioned skills to new robot settings via low-rank fine-tuning [[8]](#ref-openvla).

In π₀'s table-bussing rubric, a half score can mean that half the objects were cleared [[9]](#ref-pi0). The score does not show how much time or assistance the remaining objects require. RECAP also measures successful work per hour [[18]](#ref-recap).

#### Language-Guided Navigation

NaVILA combines current and historical images with an instruction to predict spatial language commands. On the physical robot, a separate locomotion RL policy uses LiDAR-derived terrain information to execute those commands [[21]](#ref-navila). On the R2R-CE unseen validation split, RGB navigation achieves 54% success and 49% success-weighted path length, against NaVid's 37% and 35% [[21]](#ref-navila).

Language-level motion commands allow NaVILA's route model to be paired with different locomotion controllers. The choice of visual history also affects deployment cost: the R2R-CE ablation finds little gain beyond eight sampled frames, which the physical deployment uses to limit latency [[21]](#ref-navila).

#### Supplementary Application: Virtual-World Agents

DeepMind's 2025 SIMA 2 technical report studies keyboard and mouse commands in virtual 3D worlds [[22]](#ref-sima). The Gemini-based agent reads screen images and instructions without access to privileged game-state information. The team first trains it on gameplay data and synthetic examples that connect reasoning to action, then applies online RL on tasks with verifiable rewards.

The evaluation includes held-out games and combines automatic checks with human ratings [[22]](#ref-sima). Researchers can retry the same task from a saved game state.

#### Interpreting the Evidence

[Table 1](#table-matrix-caption) summarizes the eight core studies without ranking scores across their different experimental settings. Complete-policy comparisons establish performance on shared tasks; component ablations are more useful for identifying the effect of a particular action head or reasoning target.

The application results also test different levels of execution. NaVILA's RGB route scores compare systems with different training mixtures and exclude the physical locomotion controller [[21]](#ref-navila). SIMA 2 evaluates screen-based interaction rather than contact control [[22]](#ref-sima). Helix's company demonstration documents a working hierarchy without a controlled baseline [[20]](#ref-helix).

HiRT's dynamic tests use targets at a steady 1 cm/s. The paper leaves faster object motion untested [[12]](#ref-hirt).

<div class="vla-table-scroll" role="region" aria-labelledby="table-matrix-caption" tabindex="0" markdown="1">

| Study | Integration | Setting and Key Result | Conditions of Comparison |
| --- | --- | --- | --- |
| VoxPoser | Spatial maps and planning | Five physical-robot tasks: 88% static and 70% disturbed success [[16]](#ref-voxposer). | The same tasks compare a spatial-map planner with predefined primitives. |
| RT-2 | VLM action tokens | Physical-robot generalization tests: 62% vs. RT-1's 32% [[6]](#ref-rt2). | The scenarios are shared, but pretrained models and data differ. |
| OpenVLA | Fused visual features and action tokens | Physical-robot evaluation on 29 tasks: +16.5 pp vs. RT-2-X [[8]](#ref-openvla). | Models are tested on shared tasks; their backbones and training data differ. |
| FAST | Compressed chunk tokens | One-second chunks: 140 to 28 tokens for table bussing; 700 to 53 for T-shirt folding [[10]](#ref-fast). | The tokenizer comparison applies a similar reconstruction accuracy. |
| π₀ | Flow-based action expert | Physical laundry and assembly tasks; complex-task progress scores use ten trials per task [[9]](#ref-pi0). | Model and training ablations use task-specific completion rubrics. |
| Discrete Diffusion VLA | Iterative action-token refinement | LIBERO simulation, Qwen decoder ablation: autoregressive 81.2%, parallel 94.2%, discrete 95.1% [[17]](#ref-discrete). | Heads share Qwen initialization and 30,000 training steps across four suites. |
| RT-H | Language motion before control | Eight physical-robot tasks: higher success than RT-2 on six [[15]](#ref-rth). | Label ablations share the PaLI-X backbone and compare language with classes or clusters. |
| ECoT | Scene-specific reasoning before control | Physical WidowX trials: approximately +28 pp vs. OpenVLA [[14]](#ref-ecot). | The baseline shares the VLM backbone and Bridge V2 robot data; reasoning supervision is added. |

</div>

<p id="table-matrix-caption" class="vla-caption"><strong>Table 1.</strong> Eight studies grouped by their primary integration pattern. Success-rate gains are in percentage points (pp).</p>

Success rates can conceal where an improvement occurs. Correcting an object prediction in ECoT can prevent a misdirected reach [[14]](#ref-ecot), while Liu et al.'s RL policy adjusts a reach already underway [[11]](#ref-rlvla). Separate measurements at these stages could show whether the gain comes from better object estimates or from correcting the movement. Task difficulty matters as well, since VLA-Arena finds that policy rankings change as tasks become harder [[23]](#ref-arena).

## Conclusion

RT-2 and OpenVLA transfer pretrained visual and language knowledge into action prediction [[6]](#ref-rt2), [[8]](#ref-openvla). VoxPoser connects language to control through spatial maps and a motion planner, so a constraint such as avoiding the vase can be specified explicitly [[16]](#ref-voxposer). Reliable execution still requires more than recognizing the task. FAST's compact targets reduce training cost, but its autoregressive implementation decodes more slowly than π₀ [[10]](#ref-fast). For Discrete Diffusion, the clearest advantage over parallel decoding in the reviewed tests is under language perturbations, rather than nominal task success [[17]](#ref-discrete). Even quick predictions cannot correct a reach if the controller continues following an old action prefix. RT-H and ECoT expose intermediate predictions that an operator can revise before the next action [[15]](#ref-rth), [[14]](#ref-ecot); RL adds recovery experience after execution has deviated from the intended reach [[11]](#ref-rlvla).

An incorrect object estimate may guide several subsequent actions unless the robot checks it again during a long workflow. Further training can also change behavior that previously worked, as GR00T N1's loss of a handoff after right-hand-only adaptation shows [[19]](#ref-gr00t). Recovery through human intervention consumes expert time, and autonomous physical exploration requires limits on collisions or dropped objects. Evaluating these reliability costs is difficult when studies use fragmented settings and limited trials, especially where near-saturated benchmarks leave small differences in success.

This review prioritizes task-state memory for longer workflows. The proposed record would track completed subtasks and preserve uncertain object estimates for comparison with subsequent observations. NaVILA's limited gains beyond eight history frames concern the number of sampled images [[21]](#ref-navila); explicitly recording which objects have been cleared changes the content of memory, so its benefit requires a separate test. The controller could also shorten its executed prefix when visual or tactile signals indicate unexpected contact. The difficulty lies in interrupting an unreliable sequence without disrupting stable motion. Full completion and recovery time, reported with trial-level uncertainty, would test whether these changes improve long-task execution.

## References

<ol class="vla-references">
  <li id="ref-rt1">A. Brohan et al., “RT-1: Robotics Transformer for Real-World Control at Scale,” in <em>Proceedings of Robotics: Science and Systems XIX</em>, 2023. [Online]. Available: <a href="https://doi.org/10.15607/RSS.2023.XIX.025">https://doi.org/10.15607/RSS.2023.XIX.025</a>.</li>
  <li id="ref-diffpolicy">C. Chi et al., “Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,” in <em>Proceedings of Robotics: Science and Systems XIX</em>, 2023. [Online]. Available: <a href="https://doi.org/10.15607/RSS.2023.XIX.026">https://doi.org/10.15607/RSS.2023.XIX.026</a>.</li>
  <li id="ref-survey">Y. Ma et al., “A Survey on Vision–Language–Action Models for Embodied AI,” <em>IEEE Transactions on Neural Networks and Learning Systems</em>, vol. 37, no. 7, pp. 3031–3051, 2026. [Online]. Available: <a href="https://doi.org/10.1109/TNNLS.2025.3650584">https://doi.org/10.1109/TNNLS.2025.3650584</a>.</li>
  <li id="ref-cliport">M. Shridhar, L. Manuelli, and D. Fox, “CLIPort: What and Where Pathways for Robotic Manipulation,” in <em>Proceedings of the 5th Conference on Robot Learning (CoRL 2021)</em>, PMLR, vol. 164, pp. 894–906, 2022. [Online]. Available: <a href="https://proceedings.mlr.press/v164/shridhar22a.html">https://proceedings.mlr.press/v164/shridhar22a.html</a>.</li>
  <li id="ref-palme">D. Driess et al., “PaLM-E: An Embodied Multimodal Language Model,” in <em>Proceedings of the 40th International Conference on Machine Learning</em>, PMLR, vol. 202, pp. 8469–8488, 2023. [Online]. Available: <a href="https://proceedings.mlr.press/v202/driess23a.html">https://proceedings.mlr.press/v202/driess23a.html</a>.</li>
  <li id="ref-rt2">B. Zitkovich et al., “RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control,” in <em>Proceedings of the 7th Conference on Robot Learning (CoRL 2023)</em>, PMLR, vol. 229, pp. 2165–2183, 2023. [Online]. Available: <a href="https://proceedings.mlr.press/v229/zitkovich23a.html">https://proceedings.mlr.press/v229/zitkovich23a.html</a>.</li>
  <li id="ref-octo">D. Ghosh et al., “Octo: An Open-Source Generalist Robot Policy,” in <em>Proceedings of Robotics: Science and Systems XX</em>, 2024. [Online]. Available: <a href="https://doi.org/10.15607/RSS.2024.XX.090">https://doi.org/10.15607/RSS.2024.XX.090</a>.</li>
  <li id="ref-openvla">M. J. Kim et al., “OpenVLA: An Open-Source Vision-Language-Action Model,” in <em>Proceedings of the 8th Conference on Robot Learning (CoRL 2024)</em>, PMLR, vol. 270, pp. 2679–2713, 2025. [Online]. Available: <a href="https://proceedings.mlr.press/v270/kim25c.html">https://proceedings.mlr.press/v270/kim25c.html</a>.</li>
  <li id="ref-pi0">K. Black et al., “π₀: A Vision-Language-Action Flow Model for General Robot Control,” in <em>Proceedings of Robotics: Science and Systems XXI</em>, 2025. [Online]. Available: <a href="https://doi.org/10.15607/RSS.2025.XXI.010">https://doi.org/10.15607/RSS.2025.XXI.010</a>.</li>
  <li id="ref-fast">K. Pertsch et al., “FAST: Efficient Action Tokenization for Vision-Language-Action Models,” in <em>Proceedings of Robotics: Science and Systems XXI</em>, 2025. [Online]. Available: <a href="https://doi.org/10.15607/RSS.2025.XXI.012">https://doi.org/10.15607/RSS.2025.XXI.012</a>.</li>
  <li id="ref-rlvla">J. Liu et al., “What Can RL Bring to VLA Generalization? An Empirical Study,” in <em>Advances in Neural Information Processing Systems</em>, vol. 38, pp. 107724–107754, 2025. [Online]. Available: <a href="https://doi.org/10.52202/085713-3248">https://doi.org/10.52202/085713-3248</a>.</li>
  <li id="ref-hirt">J. Zhang et al., “HiRT: Enhancing Robotic Control with Hierarchical Robot Transformers,” in <em>Proceedings of the 8th Conference on Robot Learning (CoRL 2024)</em>, PMLR, vol. 270, pp. 933–946, 2025. [Online]. Available: <a href="https://proceedings.mlr.press/v270/zhang25b.html">https://proceedings.mlr.press/v270/zhang25b.html</a>.</li>
  <li id="ref-deploysurvey">K. Kawaharazuka et al., “Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications,” <em>IEEE Access</em>, vol. 13, pp. 162467–162504, 2025. [Online]. Available: <a href="https://doi.org/10.1109/ACCESS.2025.3609980">https://doi.org/10.1109/ACCESS.2025.3609980</a>.</li>
  <li id="ref-ecot">M. Zawalski et al., “Robotic Control via Embodied Chain-of-Thought Reasoning,” in <em>Proceedings of the 8th Conference on Robot Learning (CoRL 2024)</em>, PMLR, vol. 270, pp. 3157–3181, 2025. [Online]. Available: <a href="https://proceedings.mlr.press/v270/zawalski25a.html">https://proceedings.mlr.press/v270/zawalski25a.html</a>.</li>
  <li id="ref-rth">S. Belkhale et al., “RT-H: Action Hierarchies using Language,” in <em>Proceedings of Robotics: Science and Systems XX</em>, 2024. [Online]. Available: <a href="https://doi.org/10.15607/RSS.2024.XX.049">https://doi.org/10.15607/RSS.2024.XX.049</a>.</li>
  <li id="ref-voxposer">W. Huang et al., “VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models,” in <em>Proceedings of the 7th Conference on Robot Learning (CoRL 2023)</em>, PMLR, vol. 229, pp. 540–562, 2023. [Online]. Available: <a href="https://proceedings.mlr.press/v229/huang23b.html">https://proceedings.mlr.press/v229/huang23b.html</a>.</li>
  <li id="ref-discrete">Z. Liang et al., “Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies,” in <em>Proceedings of the 43rd International Conference on Machine Learning</em>, PMLR, vol. 306, pp. 73258–73274, 2026. [Online]. Available: <a href="https://proceedings.mlr.press/v306/liang26l.html">https://proceedings.mlr.press/v306/liang26l.html</a>.</li>
  <li id="ref-recap">Physical Intelligence et al., “π<sup>*</sup><sub>0.6</sub>: A VLA That Learns From Experience,” arXiv preprint arXiv:2511.14759, 2025. [Online]. Available: <a href="https://arxiv.org/abs/2511.14759">https://arxiv.org/abs/2511.14759</a>.</li>
  <li id="ref-gr00t">NVIDIA, “GR00T N1: An Open Foundation Model for Generalist Humanoid Robots,” arXiv preprint arXiv:2503.14734, 2025. [Online]. Available: <a href="https://arxiv.org/abs/2503.14734">https://arxiv.org/abs/2503.14734</a>.</li>
  <li id="ref-helix">Figure, “Introducing Helix 02: Full-Body Autonomy,” company system report, Jan. 27, 2026. [Online]. Available: <a href="https://www.figure.ai/news/helix-02">https://www.figure.ai/news/helix-02</a>.</li>
  <li id="ref-navila">A.-C. Cheng et al., “NaVILA: Legged Robot Vision-Language-Action Model for Navigation,” in <em>Proceedings of Robotics: Science and Systems XXI</em>, 2025. [Online]. Available: <a href="https://doi.org/10.15607/RSS.2025.XXI.018">https://doi.org/10.15607/RSS.2025.XXI.018</a>.</li>
  <li id="ref-sima">SIMA Team, Google DeepMind, “SIMA 2: A Generalist Embodied Agent for Virtual Worlds,” technical report, Dec. 5, 2025. [Online]. Available: <a href="https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/SIMA_Tech_Report_2025.pdf">https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/SIMA_Tech_Report_2025.pdf</a>.</li>
  <li id="ref-arena">B. Zhang et al., “VLA-Arena: An Open-Source Framework for Benchmarking Vision-Language-Action Models,” in <em>Proceedings of the 43rd International Conference on Machine Learning</em>, PMLR, vol. 306, pp. 156567–156614, 2026. [Online]. Available: <a href="https://proceedings.mlr.press/v306/zhang26cp.html">https://proceedings.mlr.press/v306/zhang26cp.html</a>.</li>
</ol>

*helped by GPT6.1 sol*
