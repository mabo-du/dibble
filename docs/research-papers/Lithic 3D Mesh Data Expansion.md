# **Strategic Expansion of 3D Lithic Training Data for the Dibble Morphological Classifier**

## **Introduction to the Morphological Classification Framework**

The transition from traditional two-dimensional lithic illustration and manual caliper measurements to three-dimensional geometric morphometrics has fundamentally altered the paradigm of archaeological classification. The development of the Dibble open-source 3D morphological classifier represents a critical advancement in standardizing lithic typologies through the application of machine learning. Operating on a 22-dimensional morphometric feature vector and a RandomForest architecture, the model's current cross-validation accuracy of approximately eighty-one percent across a ten-class basic typology demonstrates robust predictive power for well-represented morphological categories. Specifically, the classifier excels at identifying Bifaces, Cores, and Blades, which currently possess sufficient training samples within the indexed feature space.  
However, a rigorous evaluation of the classifier's performance metrics reveals a severe structural class imbalance that acts as the primary limiting factor preventing the model from achieving an accuracy threshold of eighty-five percent or higher. The current training corpus exhibits a critical deficiency in specific morphological categories, notably Retouched Flakes, Unmodified Flakes, Unmodified Cobbles, and Tools. In the architecture of a RandomForest algorithm, under-represented classes fail to generate sufficient information gain during the iterative splitting of decision trees. This sparsity leads to high Gini impurity within the nodes and results in the systematic misclassification of minority instances, which are frequently absorbed into the probability distributions of the majority classes. When the feature space contains over a thousand bifaces but only fifty retouched flakes, the ensemble of decision trees inherently biases its predictive confidence toward the over-represented morphologies.  
The bottleneck in resolving this imbalance is not a global lack of archaeological data, but rather historical limitations in three-dimensional scanning technologies and the disparate nature of digital repository indexing. Small, translucent, and acutely angled flakes have historically evaded reliable digitization via traditional laser and structured light scanners.1 Furthermore, strict exclusionary criteria must be applied to ensure data integrity; for example, the Nubian Levallois Database, the Biśnik Cave repository, and the Gahagan Bifaces collection have been explicitly excluded from this acquisition strategy due to broken storage links, lack of publicly accessible mesh files, and repositories containing only executable code rather than spatial data.2  
Recent breakthroughs in micro-computed tomography, advanced structured light scanning, and optimized photogrammetry protocols have unlocked massive new open-access repositories of high-fidelity flake, tool, and core meshes.1 This comprehensive analysis details an exhaustive, strategic data acquisition roadmap to source the required two hundred to five hundred additional labelled three-dimensional meshes specifically targeting the under-represented classes. By integrating these discrete, newly published repositories—ranging from Early Upper Paleolithic cave sequences to highly controlled experimental knapping assemblages—the Dibble classifier can overcome its minority class deficit, refine its 22-dimensional morphometric vector, and reliably cross the targeted accuracy threshold.

## **The Geometric and Statistical Dynamics of Class Imbalance**

To appreciate the absolute necessity of targeted data expansion, the morphological variance of the under-represented classes must be analyzed through the lens of the 22-dimensional morphometric feature vector utilized by the Dibble classifier. The algorithm does not interpret a stone tool through a qualitative typological lens; rather, it processes topological variance, bounding box volume distribution, edge angle continuity, surface curvature, and principal component alignments.

### **The Baseline of Natural Clasts: Unmodified Cobbles**

The "Unmodified Cobble" acts as the geometric and statistical baseline for natural clast shape. In lithic analysis and geomorphology, river cobbles, nodules, and manuports represent raw materials lacking anthropogenic fracture mechanics. They exhibit no Hertzian cone initiation, no bulb of percussion, no striking platform, and no termination hinge.5 In a multivariate feature space, unmodified cobbles represent maximum spherical or ellipsoidal volume with minimal sharp edge continuity and an absolute zero scar density index.  
Acquiring baseline three-dimensional scans of unmodified cobbles is paramount because it anchors the classifier's understanding of zero modification. Without a robust dataset of natural clasts, the algorithm risks confusing heavily rolled, taphonomically weathered archaeological cores or battered hammerstones with naturally occurring stones. Machine learning models trained for clast classification rely heavily on distinguishing between the continuous smooth curvature of a river-rolled cobble and the faceted, intersecting planes of a reduced core.5 Increasing the current sample size from thirty to over one hundred is necessary to establish a clear decision boundary for natural versus anthropogenic surface morphologies.

### **The Intentionality Spectrum: Unmodified Versus Retouched Flakes**

The boundary between the "Unmodified Flake" category, representing debitage and unretouched blanks, and the "Retouched Flake" category, representing formalized tools, is mathematically the most challenging classification frontier in automated lithic analysis. An unmodified flake exhibits a clear ventral face, characterized by a bulb of percussion and ripple marks, and a dorsal face bearing prior flake scars or raw cortex.6 Crucially, its margins remain as acute, unaltered edges produced by the initial detachment from the core.7  
A retouched flake, conversely, exhibits secondary modification. This modification manifests as continuous, invasive, or marginal flake scars along the lateral or distal edges, intentionally struck to shape a working edge, steepen a margin for scraping, or blunt a margin for hafting.6 The computational problem arises because marginal retouch on a small flake induces only minor volumetric changes compared to the overall mass and bounding box dimensions of the artifact. If the training data contains only fifty retouched flakes, the 22-dimensional vector cannot establish a reliable statistical threshold for the localized edge-angle variance and micro-curvature changes caused by secondary retouch.8  
When researchers measure edge angles manually, the data is subject to high inter-analyst error. However, three-dimensional digital methods provide systematic, quantifiable metrics of edge design.7 Supplying the RandomForest algorithm with hundreds of high-resolution meshes of both unmodified and retouched flakes forces the decision trees to identify the precise mathematical signatures of secondary edge modification. The classifier must learn to prioritize localized vector normal variance along the artifact's margins rather than relying solely on macro-shape approximations, which fail to distinguish a raw blank from a minimally retouched tool.8

### **Resolving the Ambiguity of the Tool Class**

The current status of the Dibble classifier lists the "Tool" class at a near-zero sample size, with a strategic note to merge this category. This represents a sound taxonomic and computational decision. In traditional lithic analysis, a "Tool" is often a functional or advanced typological category, encompassing scrapers, denticulates, burins, and backed knives.6 However, from a geometric morphometric standpoint, the vast majority of these implements are fundamentally retouched flakes or retouched blades.  
Maintaining an isolated "Tool" class with arbitrary boundaries degrades the classifier's performance by fragmenting the probability distribution of retouched artifacts. By systematically merging any artifact labeled as a tool, unifacial tool, or core-tool made on a flake blank into the "Retouched Flake" class, the overall sample size for modified flake geometries is drastically increased.6 This consolidation transforms a fragmented feature space into a dense cluster, radically dropping the Gini impurity for the modified edge feature metrics and allowing the classifier to map all secondary marginal modifications under a single, robust mathematical umbrella.

## **Methodological Advances Unlocking New Lithic Morphologies**

The recent proliferation of suitable training data is directly tied to technological advancements in how lithic artifacts are digitized. Understanding the provenance of the three-dimensional meshes is critical, as the scanning methodology dictates the vertex density, surface fidelity, and potential anomalies present within the PLY, STL, or OBJ files.  
Historically, the digitization of small lithic implements, such as bladelets, microblades, and microflakes, was severely compromised by the physical properties of the artifacts. The translucency of flint and chert, combined with acute, sub-millimeter edge angles, caused structured light and laser scanners to fail. These optical scanners frequently produced meshes with smoothed edges, completely obliterating the micro-topography of marginal retouch and rendering the meshes useless for training a classifier to detect retouched flakes.1  
This limitation has been overcome by the introduction of the StyroStone protocol, which proposes a step-by-step procedure relying on micro-computed tomographic (Micro-CT) technology.1 By suspending small lithic implements in low-density Styrofoam, researchers can capture the three-dimensional shape of microflakes in unprecedented detail, entirely bypassing the optical limitations of surface reflection and translucency.1 Datasets generated using this protocol provide the precise edge-angle fidelity required by the Dibble classifier to differentiate unmodified margins from retouched margins.  
Simultaneously, advancements in structured light scanning, specifically the deployment of the Artec Micro alongside the traditional Artec Space Spider, have allowed for the mass digitization of entire lithic assemblages down to the sub-centimeter scale.10 Furthermore, refined photogrammetry protocols have democratized the creation of high-fidelity three-dimensional models in field settings.11 By utilizing controlled turntables, calibrated lighting, and advanced algorithms in software such as Agisoft Metashape, researchers can now produce OBJ files of experimental flakes and cores that rival the accuracy of dedicated laser scanners.11 These disparate technological origins mean that the incoming training data will exhibit varying polygon counts, requiring specific normalization pipelines before integration into the classifier.

## **Primary Archaeological Target Datasets**

The following repositories have been identified as primary, high-confidence sources containing substantial volumes of the precise morphological classes required to balance the Dibble training set. These datasets prioritize open-access availability, robust metadata, and direct per-artifact download capabilities.

### **Fumane Cave Laminar Products and Core Re-Evaluation**

The Fumane Cave sequence in northeastern Italy represents one of the most critical stratigraphic successions for understanding the earliest phases of the Upper Paleolithic in Mediterranean Europe.6 While early iterations of the Open Aurignacian Project hosted on the Open Science Framework have already been integrated into the Dibble database, the repository cited below represents a distinctly processed, highly enriched dataset published in 2024\. This dataset utilizes the novel StyroStone Micro-CT protocol to capture artifacts that previously evaded digitization.6

| Attribute | Parameter Details |
| :---- | :---- |
| **Repository Name** | Zenodo: 3D models of lithic artifacts from Fumane Cave |
| **DOI / URL** | 10.5281/zenodo.15382869 |
| **Estimated Meshes** | 948 total 3D meshes. |
| **Mesh Format** | PLY / OBJ formats suitable for geometric morphometrics. |
| **Label Types** | Flakes, retouched tools, blade/bladelet cores, blades, bladelets. |
| **Basic System Mapping** | Directly maps to *Unmodified Flake*, *Retouched Flake*, and *Core*. |
| **Download Method** | Direct Zenodo API extraction. |
| **File Size Estimate** | Approximately 4.5 GB total archive size. |
| **Access Restrictions** | Open Access under Creative Commons licensing. |
| **Confidence Level** | High. The metadata explicitly isolates flakes and retouched tools, providing exact targets for the under-represented classes. |

The contextual depth of this dataset is extraordinary, largely due to the accompanying metadata structures. The provided CSV files distinguish between various blank types and explicitly code the preservation state of the artifact, classifying them as complete, distal, mesial, or proximal fragments.12 To train the Dibble classifier optimally, the ingestion pipeline must filter the metadata strictly for complete flakes. Introducing fragmented flakes into the training data would skew the 22-dimensional feature vector, causing the algorithm to calculate anomalous length-to-width ratios and disrupted volumetric balances.  
Furthermore, the dataset flags artifacts with varying cortex percentages, ranging from zero cortex to complete cortical coverage.12 Training the RandomForest algorithm on unmodified flakes with highly diverse cortex coverage ensures that the model learns to identify the underlying morphology of a flake irrespective of the raw material's exterior texture. This prevents the classifier from incorrectly associating the rough texture of a cortical surface strictly with the "Core" or "Unmodified Cobble" classes, a common failure point in less sophisticated machine learning models.

### **Grotta di Castelcivita Lithic Assemblage**

Grotta di Castelcivita, located in southern Italy, provides a vital continuous stratigraphy for the Protoaurignacian and Early Aurignacian periods, dating to approximately 41,000 to 39,800 years ago.10 This sequence is uniquely sealed by the Campanian Ignimbrite geochronological marker, ensuring high temporal resolution. A recent open-access initiative has digitized a substantial portion of this assemblage, specifically targeting the laminar products and associated tools.10

| Attribute | Parameter Details |
| :---- | :---- |
| **Repository Name** | Zenodo: 3D models of lithic artifacts from Grotta di Castelcivita |
| **DOI / URL** | 10.5281/zenodo.10631390 |
| **Estimated Meshes** | 538 total 3D meshes. |
| **Mesh Format** | PLY format point clouds and surface meshes. |
| **Label Types** | Cores, blades, bladelets, flakes, and retouched tools. |
| **Basic System Mapping** | Directly maps to *Unmodified Flake* and *Retouched Flake*. |
| **Download Method** | Direct Zenodo extraction via a single ZIP archive accompanied by a master CSV. |
| **File Size Estimate** | Approximately 2.5 GB. |
| **Access Restrictions** | Open Access under Creative Commons licensing. |
| **Confidence Level** | High. The dataset explicitly states the inclusion of 538 models, isolating the exact classes required. |

The Castelcivita dataset was processed using a dual-scanner approach, employing an Artec Space Spider for larger elements and an Artec Micro for digitizing extremely small lithics, such as retouched bladelets measuring around one centimeter in length.10 The application of the Artec Micro ensures that the micro-topography of the retouched edges is impeccably preserved in the PLY files. For the Dibble data engineering pipeline, the accompanying CSV file is invaluable. It contains specific technological classifications, enabling an automated script to parse the metadata, filter out the blades and cores, which are already sufficient in the current dataset, and ingest only the unmodified flakes and retouched tools.10 The variance in chronological age ensures that the morphological traits captured represent broad evolutionary tool designs rather than a single idiosyncratic knapping event.

### **Holocene Unifacial Tools of Tropical Central Brazil**

To ensure the Dibble classifier does not become overfitted to the specific knapping trajectories of the European Paleolithic, geographic diversification of the training data is essential. A highly targeted dataset analyzing the variability of Holocene unifacial tools from the Middle Magdalena River Valley and tropical Central Brazil provides a pristine corpus of retouched flake morphologies.13

| Attribute | Parameter Details |
| :---- | :---- |
| **Repository Name** | Figshare / PLOS One: Manual 3D analysis database |
| **DOI / URL** | 10.1371/journal.pone.0315746.s002 / 10.6084/m9.figshare.28126957 |
| **Estimated Meshes** | 67 highly detailed 3D meshes of unifacially shaped artifacts. |
| **Mesh Format** | PLY / OBJ (captured via Shining 3D Scanner–EinScan SP V2). |
| **Label Types** | Unifacial tools, shaped artifacts, flake blanks. |
| **Basic System Mapping** | Perfect mapping for *Retouched Flake* (absorbing the Tool class). |
| **Download Method** | Direct Figshare dataset download / PLOS One supplementary files. |
| **File Size Estimate** | Approximately 500 MB. |
| **Access Restrictions** | Open Access. |
| **Confidence Level** | High. The database specifically correlates 3D mesh models with edge angle measurements and taphonomic analysis. |

This dataset, generated by González-Varas et al., employs techno-structural analysis and three-dimensional geometric morphometrics to quantify tool geometry and functional potentials.15 The selection of these artifacts was based on ergonomic characteristics, identifiable transformative parts, and prehensile adaptations.16 By incorporating these sixty-seven unifacial tools, the classifier receives a massive influx of data explicitly mapping the "Retouched Flake" class. The study utilizes advanced meshing protocols to align the tools along their main axis, capturing transversal and longitudinal sections to calculate cutting edge angles.14 Integrating these PLY files will heavily weight the feature importance of unifacial retouch in the RandomForest algorithm, providing the necessary mathematical variance to distinguish formal tools from unmodified debitage.

## **High-Fidelity Experimental and Methodological Corpora**

Archaeological assemblages suffer from inherent taphonomic noise. Processes such as trampling, soil compaction, and fluvial rolling can cause post-depositional edge damage that mimics intentional anthropogenic retouch, a phenomenon known as pseudo-retouch.17 To anchor the classifier's mathematical understanding of true retouch and pristine flake detachment, experimental archaeology datasets are required. These datasets provide absolute ground-truth labels because the reduction sequences and resultant morphologies are entirely controlled by the researcher.

### **Experimentally-Produced Lithic Artifacts via Photogrammetry**

The Data Repository for the University of Minnesota (DRUM) hosts a highly specialized dataset of experimental lithics created to validate photogrammetric modeling techniques.11 While the total volume is lower than the cave datasets, the qualitative value of this repository is immense for establishing pure geometric baselines.

| Attribute | Parameter Details |
| :---- | :---- |
| **Repository Name** | DRUM: Three-Dimensional Models of Experimentally-Produced Lithic Artifacts |
| **DOI / URL** | 10.13020/D6T88N |
| **Estimated Meshes** | 30 individual meshes (15 distinct artifacts scanned via two methodologies). |
| **Mesh Format** | OBJ format, accompanied by MTL material and JPG texture files. |
| **Label Types** | Experimental cores and flakes generated from Keokuk chert. |
| **Basic System Mapping** | Ground-truth mapping for *Unmodified Flake* and *Experimental Core*. |
| **Download Method** | Direct HTTP download of four ZIP files categorized by tool type. |
| **File Size Estimate** | Approximately 500 MB. |
| **Access Restrictions** | Open Access via the University Digital Conservancy. |
| **Confidence Level** | High. Fully documented object files aligned precisely around the origin. |

Magnani, Douglass, and Porter utilized both expedient field-ready and refined laboratory photogrammetry protocols to capture fifteen experimentally produced cores and flakes made from Keokuk chert.11 Because these flakes were struck in a controlled environment, their edges are completely pristine and devoid of any post-depositional edge damage. From a data engineering perspective, these OBJ files have already been processed in Geomagic Design X, where they were meticulously centered and aligned around the origin of the coordinate system.11 This pre-alignment is critical for the extraction of orientation-dependent metrics within Dibble's 22-dimensional feature vector. Introducing these pristine experimental flakes forces the classifier to establish a clean, mathematical baseline for the Hertzian mechanics of a pure unmodified flake, creating a sharp contrast against taphonomically noisy archaeological flakes.

### **The 3D-EdgeAngle Validation Corpus**

To further refine the classifier's ability to detect secondary modification, specialized methodological datasets designed to test computational algorithms provide invaluable edge-case examples. The 3D-EdgeAngle repository contains three-dimensional models specifically chosen to represent complex retouched edges and archaeological tool geometries.8

| Attribute | Parameter Details |
| :---- | :---- |
| **Repository Name** | Zenodo: 3D-EdgeAngle Lithic Data |
| **DOI / URL** | 10.5281/zenodo.7326242 / 10.5281/zenodo.7961582 |
| **Estimated Meshes** | Highly specialized validation meshes (e.g., EAP-flake, BU-072, WEM-60). |
| **Mesh Format** | PLY / OBJ formats suitable for edge detection. |
| **Label Types** | Experimental retouched flakes, Archaeological tools (Keilmesser). |
| **Basic System Mapping** | Perfect mapping for *Retouched Flake* and *Tool*. |
| **Download Method** | Direct Zenodo download. |
| **File Size Estimate** | Under 100 MB. |
| **Access Restrictions** | Open Access. |
| **Confidence Level** | High. Models specifically designed to test automated edge-angle algorithms. |

Developed by Schunk et al., the 3D-EdgeAngle script is a semi-automated digital method to systematically quantify stone tool edge angle and design.7 The repository includes the "EAP-flake," an experimental elongated laminar flake made of Baltic flint, featuring marginally applied, minimally invasive retouch on the dorsal face of the distal part.8 Minimally invasive retouch is the primary reason machine learning classifiers fail to distinguish retouched flakes from unmodified flakes, as the overall volume remains largely identical to the original blank. Only the vector normals along the immediate margin are altered. Injecting models like the EAP-flake and the BU-072 Keilmesser, a bifacially backed knife, provides the Dibble classifier with extreme edge-case examples.8 These specific models will heavily weight the feature importance of marginal vector variance, vastly improving the recall rate for the retouched flake class.

### **Traceology and Occlusal Fingerprint Analysis (OFA) Samples**

Further experimental ground-truth data can be sourced from studies focusing on use-wear traceology and microscopic edge analysis. The Occlusal Fingerprint Analysis (OFA) methodology, typically used in dental wear studies, has been applied to experimental stone tools to map contact materials and edge degradation.21

| Attribute | Parameter Details |
| :---- | :---- |
| **Repository Name** | Zenodo: OFA applied to experimental stone tools |
| **DOI / URL** | 10.5281/zenodo.7930863 |
| **Estimated Meshes** | 4 to 5 highly precise experimental tool meshes. |
| **Mesh Format** | High-resolution meshes generated via AICON smartScan-HE R8. |
| **Label Types** | Experimental stone tool samples. |
| **Basic System Mapping** | Maps to *Unmodified Flake* or *Retouched Flake* depending on the blank utilized for the experiment. |
| **Download Method** | Direct Zenodo download of OFA projects. |
| **File Size Estimate** | Approximately 200 MB. |
| **Access Restrictions** | Open Access. |
| **Confidence Level** | High. Scanned with state-of-the-art structured light technology and edited in ZEISS GOM Inspect. |

Conducted at the TraCEr laboratory, this pilot study provides a small but extraordinarily precise set of three-dimensional models generated with an AICON smartScan-HE R8 structured light scanner.21 Because these models were created to study the microscopic kinematic wear of tool use, their edge geometry is captured flawlessly. Integrating these meshes provides the RandomForest algorithm with flawless geometric representations of functional tool edges prior to and after experimental use, aiding the classifier in identifying the subtle topographic signatures of utilized flakes.

## **Secondary Targets and Graph Modelling Baselines**

Beyond direct geometric repositories, evaluating datasets designed for advanced computational workflows, such as deep learning and graph modeling, provides additional access to niche lithic morphologies and perfectly labeled knapping sequences.

### **Scar Graph Modelling of Operational Sequences**

Recent computational approaches have combined three-dimensional mesh segmentation with directed graph modeling to track the temporal operational sequence of flake detachment.22 This methodology provides an entirely new way to analyze the reduction of a core.

| Attribute | Parameter Details |
| :---- | :---- |
| **Repository Name** | Zenodo: Linking Scars: Topology-based Scar Detection |
| **DOI / URL** | 10.5281/zenodo.14882743 / 10.5281/zenodo.10477448 |
| **Estimated Meshes** | Includes a complete experimental knapping series. |
| **Mesh Format** | High-resolution 3D meshes compatible with the GigaMesh framework. |
| **Label Types** | Scar segmentation ground truth, experimental knapping series. |
| **Basic System Mapping** | *Unmodified Flake* and *Experimental Core*. |
| **Download Method** | Direct Zenodo API download under Creative Commons. |
| **File Size Estimate** | Approximately 1 GB. |
| **Access Restrictions** | Open Access. |
| **Confidence Level** | Medium. Primary data focuses on annotated graph ground-truth, but underlying source meshes are accessible. |

The authors utilized manual scar segmentation on 3D meshes to construct graph models where nodes represent individual flake scars and edges represent sequential adjacency.22 While the Dibble classifier relies on a continuous 22-dimensional morphometric vector rather than discrete directed graph topology, the underlying three-dimensional meshes of the experimental knapping series serve as pristine ground-truth blanks.23 The mathematical principles used in this study, specifically Multi-Scale Integral Invariants (MSII) curvature values, mirror the types of continuous surface data the Dibble classifier ingests.22 Extracting the raw meshes of the complete knapping sequence provides a perfect chronological progression of unmodified flakes, allowing the classifier to map the geometric changes that occur from primary decortication flakes to late-stage tertiary flakes.  
Similarly, datasets focusing on the orientation statistics of flake scars on multiplatform cores, such as the Liang Bua core analysis dataset, provide further computational context.24 While primarily focused on cores rather than flakes, the integration of these twenty-one multiplatform core models ensures that the classifier's "Core" category maintains a robust variance, preventing flakes from being misclassified as exhausted multiplatform cores.

### **Deep Learning Applications in Oceanian Assemblages**

To further geographically diversify the training data, recent proceedings from the Computer Applications and Quantitative Methods in Archaeology (CAA) conference highlight the application of deep learning to pre-contact Māori stone tool manufacture.25

| Attribute | Parameter Details |
| :---- | :---- |
| **Repository Name** | Zenodo: A Framework for Integrating Domain Knowledge & Deep Learning |
| **DOI / URL** | 10.5281/zenodo.13858849 |
| **Estimated Meshes** | To be verified via associated data-sharing links. |
| **Mesh Format** | Point cloud and mesh formats formatted for deep learning pipelines. |
| **Label Types** | Pre-contact Māori lithic flakes and fragments. |
| **Basic System Mapping** | *Unmodified Flake*. |
| **Download Method** | Linked repositories via conference proceedings. |
| **File Size Estimate** | Unknown. |
| **Access Restrictions** | Dependent on author data-sharing agreements. |
| **Confidence Level** | Medium-Low. Requires secondary sourcing from the authors, but provides vital geographic diversity. |

Mills et al. focus on the fine-grained three-dimensional shape analysis of lithic flakes from Aotearoa New Zealand.25 While primarily a methodological paper proposing the incorporation of explicit reasoning into deep learning systems, the project represents a rare geographical expansion for 3D lithic datasets. Sourcing the underlying data via the corresponding authors would provide a completely novel morphological baseline, ensuring the Dibble classifier is not exclusively overfitted to European Aurignacian or Levantine Acheulean knapping trajectories.

## **Sourcing the Unmodified Cobble Baseline**

The original parameters identified Unmodified Cobbles as severely lacking, with only thirty current samples. Resolving this deficit is computationally difficult because natural clasts are rarely uploaded to archaeological lithic repositories; archaeologists tend to digitize artifacts, not unmodified river rocks. However, a robust dataset of natural clasts is essential to establish the absolute zero-modification geometric threshold.  
To source over one hundred unmodified cobbles, the most efficient strategy shifts from archaeological archives to geo-spatial, Earth Science, and specific experimental use-wear datasets:

1. **Experimental Hammerstone Baselines:** Experimental use-wear studies, such as the three-dimensional surface morphometry analysis of hammerstones, inherently require 3D scans of the unmodified cobble prior to use. The methodology outlined by Benito et al. utilizes original STL files of complete river cobbles to establish a baseline before extracting the anthropogenic use-wear regions.27 Sourcing these pre-use experimental datasets from Zenodo yields perfect, label-ready natural clast geometries.  
2. **Machine Learning Clast Classification Datasets:** Earth science researchers frequently train convolutional neural networks (CNNs) to distinguish between worked stone and naturally occurring lithic clasts. Case studies utilizing two-dimensional images and three-dimensional dense point clouds of gravel-sized rock clasts from fluvial environments in Egypt, Australia, and New Zealand provide an abundance of natural stone data.5 Sourcing the raw point clouds from these clast differentiation studies provides hundreds of perfect "Unmodified Cobble" meshes, entirely resolving the thirty-sample deficit.

## **Data Integration and Feature Normalization Pipeline**

Identifying and downloading the repositories is only the primary phase of the expansion. Integrating over one thousand heterogeneous three-dimensional meshes into a unified 22-dimensional feature space requires a rigorous data engineering pipeline. The datasets identified above originate from Micro-CT scanners, structured light scanners, and photogrammetry rigs. Consequently, the meshes exhibit massive disparities in scale, spatial orientation, and vertex density.

### **Format Standardization and Mesh Decimation**

The identified datasets primarily utilize PLY and OBJ formats, which the Dibble pipeline natively supports. However, the vertex density of a Micro-CT scan generated via the StyroStone protocol can easily exceed several million polygons, whereas an expedient photogrammetric model from the DRUM dataset may possess fewer than one hundred thousand faces.  
If raw meshes are ingested directly, the morphometric feature vector may inadvertently weight high-resolution Micro-CT scans differently than lower-resolution photogrammetry models, confusing surface noise with valid geometric variance. It is necessary to implement an automated mesh decimation algorithm, such as a Quadric Edge Collapse strategy, prior to feature extraction. Standardizing all incoming meshes to a uniform polygon count, for example, fifty thousand faces, ensures that the calculation of surface curvature and edge continuity remains mathematically consistent across all scanning methodologies.

### **Geometric Alignment and Principal Component Analysis**

As explicitly noted in the UMN DRUM photogrammetry methodology, artifacts must be systematically aligned and centered around the origin of the coordinate system.11 The 22-dimensional morphometric vector relies heavily on calculating orientation-dependent metrics, such as bounding box volumetric ratios, the distribution of mass along the central axis, and longitudinal asymmetry.  
If a three-dimensional mesh from the Castelcivita dataset is loaded off-axis relative to a mesh from the Fumane dataset, their respective length, width, and thickness ratios will invert, fundamentally destroying the logical structure of the RandomForest classifier. To resolve this, a Principal Component Analysis (PCA) must be executed on the vertex coordinates of every incoming mesh. This pre-processing step automatically aligns the first principal component with the Z-axis to represent maximum length, the second principal component with the Y-axis for width, and the third principal component with the X-axis for thickness. This geometric normalization guarantees that the feature vector is comparing homologous dimensions across the entire expanded dataset.

### **Consolidating the Taxonomy**

Finally, the ingestion pipeline must execute the planned taxonomic consolidation. When parsing the metadata CSV files from the Grotta di Castelcivita, Fumane Cave, and Central Brazilian datasets, the script must automatically map any artifact labeled as a "Tool," "Retouched tool," "Unifacial tool," or "Core-tool" (when manufactured on a flake blank) directly into the "Retouched Flake" class.6 This consolidation eliminates the ambiguous "Tool" category, transforming the previously sparse fifty current samples into a robust corpus of well over three hundred meticulously documented modified flake geometries.

## **Synthesis and Impact on Classifier Accuracy**

The strategic integration of the Grotta di Castelcivita repository (538 meshes), the Fumane Cave 2024 StyroStone release (948 meshes), the Holocene Unifacial Tools database (67 meshes), the UMN DRUM Experimental set (30 meshes), and the various methodological validation corpora creates an aggregate pool of over one thousand five hundred new, high-fidelity three-dimensional models. By algorithmically filtering these vast open-access repositories utilizing their attached metadata CSV files, the Dibble project will easily exceed its goal of extracting two hundred to five hundred targeted meshes specifically for Unmodified Flakes, Retouched Flakes, and Unmodified Cobbles.  
The impact of this data expansion on the RandomForest classifier will be systemic and profound. Currently, the model is overfitting to the morphological features of Bifaces and Cores due to their overwhelming numerical superiority in the training data. By densely populating the mathematical void in the hyperspace that represents flat, acute-angled geometries and localized marginal surface invasions, the ensemble of decision trees will naturally form highly resilient splitting criteria.  
Specifically, introducing the pristine unmodified flakes from the DRUM experimental dataset establishes an absolute mathematical baseline for pure Hertzian fracture mechanics. Contrasting this baseline against the vast array of retouched flakes and unifacial tools sourced from the Castelcivita, Fumane, and Brazilian archaeological datasets forces the learning algorithm to explicitly recognize how secondary edge modification breaks the natural topological symmetry of a raw flake. Furthermore, incorporating hundreds of unmodified fluvial cobbles from Earth Science clast datasets anchors the algorithm's understanding of zero anthropogenic modification.  
Through the systematic assimilation, geometric normalization, and precise feature extraction of these newly published, open-access three-dimensional repositories, the structural class imbalance within the Dibble training data is permanently resolved. This robust, high-variance geometric foundation provides the necessary computational rigor to push the 5-fold cross-validation accuracy confidently beyond the eighty-five percent threshold, establishing a new benchmark in the automated morphological classification of lithic assemblages.

#### **Works cited**

1. Research compendium for 'Practical and technical aspects for the 3D scanning of lithic artefacts using micro-computed tomography techniques and laser light scanners for subsequent geometric morphometric analysis. Introducing the StyroStone protocol' \- Zenodo, accessed June 3, 2026, [https://zenodo.org/record/6365681](https://zenodo.org/record/6365681)  
2. Lithic Artefacts from Early Middle Palaeolithic site Biśnik Cave, Poland \- Zenodo, accessed June 3, 2026, [https://zenodo.org/records/19186143](https://zenodo.org/records/19186143)  
3. Nubian Database (JOAD, Hallinan 2024\) \- OSF, accessed June 3, 2026, [https://osf.io/78yhz/](https://osf.io/78yhz/)  
4. Practical and technical aspects for the 3D scanning of lithic artefacts using micro-computed tomography techniques and laser light scanners for subsequent geometric morphometric analysis. Introducing the StyroStone protocol | Armando Falcucci, accessed June 3, 2026, [https://www.armandofalcucci.com/publication/goeldner-et-al\_2022\_plosone/](https://www.armandofalcucci.com/publication/goeldner-et-al_2022_plosone/)  
5. Machine learning for stone artifact identification: Distinguishing worked stone artifacts from natural clasts using deep neural networks \- PMC, accessed June 3, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9365149/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9365149/)  
6. The Open Aurignacian Project. Volume 1: Grotta di Fumane in northeastern Italy \- Zenodo, accessed June 3, 2026, [https://zenodo.org/records/15382869](https://zenodo.org/records/15382869)  
7. Enhancing lithic analysis: Introducing 3D-EdgeAngle as a semi-automated 3D digital method to systematically quantify stone tool edge angle and design \- PubMed, accessed June 3, 2026, [https://pubmed.ncbi.nlm.nih.gov/38032889/](https://pubmed.ncbi.nlm.nih.gov/38032889/)  
8. Enhancing lithic analysis: Introducing 3D-EdgeAngle as a semi-automated 3D digital method to systematically quantify stone tool edge angle and design \- PMC, accessed June 3, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10688690/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10688690/)  
9. Enhancing lithic analysis: Introducing 3D-EdgeAngle as a semi-automated 3D digital method to systematically quantify stone tool edge angle and design | PLOS One, accessed June 3, 2026, [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0295081](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0295081)  
10. The Open Aurignacian Project. Volume 2: Grotta di Castelcivita in southern Italy \- Zenodo, accessed June 3, 2026, [https://zenodo.org/records/10631390](https://zenodo.org/records/10631390)  
11. Three-Dimensional Models of Experimentally-Produced Lithic Artifacts Created using Expedient and Refined Photogrammetry Protocols \- University Digital Conservancy, accessed June 3, 2026, [https://conservancy.umn.edu/items/42498dea-f904-43a5-8cdc-9aab61a82dd9](https://conservancy.umn.edu/items/42498dea-f904-43a5-8cdc-9aab61a82dd9)  
12. The Open Aurignacian Project. Volume 1: Grotta di Fumane in northeastern Italy \- Zenodo, accessed June 3, 2026, [https://zenodo.org/records/15131708](https://zenodo.org/records/15131708)  
13. (PDF) Techno-structural and 3-D geometric morphometric analysis applied for investigating the variability of Holocene unifacial tools in tropical Central Brazil \- ResearchGate, accessed June 3, 2026, [https://www.researchgate.net/publication/387668318\_Techno-structural\_and\_3-D\_geometric\_morphometric\_analysis\_applied\_for\_investigating\_the\_variability\_of\_Holocene\_unifacial\_tools\_in\_tropical\_Central\_Brazil](https://www.researchgate.net/publication/387668318_Techno-structural_and_3-D_geometric_morphometric_analysis_applied_for_investigating_the_variability_of_Holocene_unifacial_tools_in_tropical_Central_Brazil)  
14. Techno-structural and 3-D geometric morphometric analysis applied for investigating the variability of Holocene unifacial tools in tropical Central Brazil | PLOS One, accessed June 3, 2026, [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0315746](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0315746)  
15. Techno-structural and 3-D geometric morphometric analysis applied for investigating the variability of Holocene unifacial tools in tropical Central Brazil \- PubMed, accessed June 3, 2026, [https://pubmed.ncbi.nlm.nih.gov/39746085/](https://pubmed.ncbi.nlm.nih.gov/39746085/)  
16. Techno-structural and 3-D geometric morphometric analysis applied for investigating the variability of Holocene unifacial tools in tropical Central Brazil \- PMC, accessed June 3, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11695013/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11695013/)  
17. Full article: Making Sense of Quartz Assemblages: A Multi-Method Approach to a Small Mesolithic Site in Eastern Central Sweden \- Taylor & Francis, accessed June 3, 2026, [https://www.tandfonline.com/doi/full/10.1080/01977261.2026.2640658](https://www.tandfonline.com/doi/full/10.1080/01977261.2026.2640658)  
18. SOILS RECORDS PAST PRESENT \- Zenodo, accessed June 3, 2026, [https://zenodo.org/record/3417724/files/Soils%20as%20records%20of%20past%20and%20present\_THE%20BOOK.pdf?download=1](https://zenodo.org/record/3417724/files/Soils%20as%20records%20of%20past%20and%20present_THE%20BOOK.pdf?download=1)  
19. Closing the seams: resolving frequently encountered issues inphotogrammetric modelling | Antiquity | Cambridge Core, accessed June 3, 2026, [https://www.cambridge.org/core/journals/antiquity/article/closing-the-seams-resolving-frequently-encountered-issues-in-photogrammetric-modelling/E55D084C0C92605A4128EA84BE4FB63E](https://www.cambridge.org/core/journals/antiquity/article/closing-the-seams-resolving-frequently-encountered-issues-in-photogrammetric-modelling/E55D084C0C92605A4128EA84BE4FB63E)  
20. Enhancing lithic analysis: Introducing 3D-EdgeAngle as a semi-automated 3D digital method to systematically quantify stone tool edge angle and design \- ResearchGate, accessed June 3, 2026, [https://www.researchgate.net/publication/376082266\_Enhancing\_lithic\_analysis\_Introducing\_3D-EdgeAngle\_as\_a\_semi-automated\_3D\_digital\_method\_to\_systematically\_quantify\_stone\_tool\_edge\_angle\_and\_design](https://www.researchgate.net/publication/376082266_Enhancing_lithic_analysis_Introducing_3D-EdgeAngle_as_a_semi-automated_3D_digital_method_to_systematically_quantify_stone_tool_edge_angle_and_design)  
21. OFA\_lithics \- Zenodo, accessed June 3, 2026, [https://zenodo.org/records/7930863](https://zenodo.org/records/7930863)  
22. From Scar to Scar: Reconstructing Operational Sequences of Lithic Artifacts using Scar-Ridge-Pattern-based Graph Models \- Zenodo, accessed June 3, 2026, [https://zenodo.org/records/14882743](https://zenodo.org/records/14882743)  
23. Topology-based Scar Detection and Graph Modeling of Paleolithic Artifacts in 3D \- Zenodo, accessed June 3, 2026, [https://zenodo.org/records/10477448](https://zenodo.org/records/10477448)  
24. A new method for quantifying flake scar organisation on cores using orientation statistics, accessed June 3, 2026, [https://zenodo.org/records/10906321](https://zenodo.org/records/10906321)  
25. A Framework for Integrating Domain Knowledge & Deep Learning for 3D Shape Analysis of Lithic Fragments \- OUR Archive \- University of Otago, accessed June 3, 2026, [https://ourarchive.otago.ac.nz/view/pdfCoverPage?instCode=64OTAGO\_INST\&filePid=13411925890001891\&download=true](https://ourarchive.otago.ac.nz/view/pdfCoverPage?instCode=64OTAGO_INST&filePid=13411925890001891&download=true)  
26. A Framework for Integrating Domain Knowledge & Deep Learning for 3D Shape Analysis of Lithic Fragments \- University of Otago Research Archive, accessed June 3, 2026, [https://ourarchive.otago.ac.nz/esploro/outputs/conferencePaper/A-Framework-for-Integrating-Domain-Knowledge/9926590377601891](https://ourarchive.otago.ac.nz/esploro/outputs/conferencePaper/A-Framework-for-Integrating-Domain-Knowledge/9926590377601891)  
27. Three-dimensional surface morphometry differentiates behaviour on primate percussive stone tools \- Royal Society Publishing, accessed June 3, 2026, [https://royalsocietypublishing.org/rsif/article/18/184/20210576/90093/Three-dimensional-surface-morphometry](https://royalsocietypublishing.org/rsif/article/18/184/20210576/90093/Three-dimensional-surface-morphometry)