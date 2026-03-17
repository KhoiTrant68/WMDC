# 3. Proposed Method

Our proposed architecture, WMDC, consists of three core innovations: (1) a **Frequency-Disentangled Mamba (FDM)** module for linear-time global context modeling, (2) a **Spatially-Adaptive Entropic Optimal Transport (EOT) Attention** mechanism mapping spatial priors to an image-adaptive dynamic dictionary, and (3) a **Markovian Slice-Based Context Model** with Latent Residual Prediction (LRP). 

## 3.1. Overall Architecture
Given an input image $x \in \mathbb{R}^{3 \times H \times W}$, the main encoder $g_a$ extracts a latent representation $y = g_a(x)$. To capture spatial dependencies, a hyper-encoder $h_a$ transforms $y$ into a hyperprior $z = h_a(y)$. The hyperprior is quantized and transmitted as $\hat{z}$. During decoding, $\hat{z}$ is used to predict the Gaussian distribution parameters for $y$. To exploit intra-latent dependencies, $y$ is split along the channel dimension into $K=5$ slices: $y = [y_1, y_2, \dots, y_K]$. Each slice $y_k$ is encoded sequentially, utilizing the hyperprior $\hat{z}$ and a Markovian memory state $M_k$ aggregated from previously decoded slices $\hat{y}_{<k}$. 

## 3.2. Frequency-Disentangled Mamba (FDM)
Standard Visual State Space Models (VSSMs) excel at capturing long-range structural dependencies but often struggle with high-frequency spatial details (e.g., edges and textures) due to their sequential scanning nature. To address this, we propose the **Frequency-Disentangled Mamba (FDM)** block. 

By applying a 2D Haar Discrete Wavelet Transform (DWT), we isolate the low-frequency sub-band $f_{LL}$ and route it through the Mamba block. This reduces the spatial sequence length by a factor of 4, yielding a **$4\times$ reduction in computational complexity** compared to standard VSSMs. Simultaneously, the high-frequency sub-bands $f_H$ generate affine modulation parameters ($\gamma, \beta$) that inject local textures back into the global structural representation via a Feature-wise Linear Modulation (FiLM):
\begin{equation}
    f'_{LL} = \text{VSSBlock}(f_{LL}) \odot (1 + \gamma(f_H)) + \beta(f_H)
\end{equation}

## 3.3. Image-Adaptive EOT Dictionary Attention
To accurately predict the distribution of the latent slices, we generate an **Image-Adaptive Dynamic Dictionary** $\mathcal{D} \in \mathbb{R}^{N_d \times d_{dim}}$ by cross-attending a set of learned queries with the quantized hyperprior $\hat{z}$. Standard softmax attention frequently suffers from "index collapse" (utilizing only a small subset of the dictionary), leading to sub-optimal bitrate allocation.

To enforce robust dictionary utilization, we frame the spatial-to-dictionary mapping as an **Unbalanced Entropic Optimal Transport (EOT)** problem. Let $Q \in \mathbb{R}^{(HW) \times d_{dim}}$ be the flattened spatial queries. Let $K, V \in \mathbb{R}^{N_d \times d_{dim}}$ be the keys and values from the dynamic dictionary. We define the cost matrix $C_{i,j} = 1 - \langle \bar{q}_i, \bar{k}_j \rangle$. 

Because images are highly heterogeneous, forcing strict marginal constraints (balanced optimal transport) leads to inefficient mass splitting. Therefore, we adopt an **Unbalanced Optimal Transport** formulation (Wasserstein-Fisher-Rao). We solve for a global transport plan, but dynamically scale the transport sharpness using a spatially-adaptive temperature $\epsilon_i \in \mathbb{R}^{HW}$ predicted from the hyperprior.

\begin{algorithm}[H]
\caption{Spatially-Adaptive Unbalanced Sinkhorn EOT}
\textbf{Input:} Cost matrix $C$, spatial temperature $\epsilon \in \mathbb{R}^{HW}$, relaxation $\tau$, iterations $T=3$\\
\textbf{Output:} Transport Plan $P \in \mathbb{R}^{HW \times N_d}$

\begin{algorithmic}[1]
\STATE $\bar{\epsilon} \gets \text{mean}(\epsilon)$ \quad \textit{// Global epsilon for solver convergence}
\STATE $u \gets \mathbf{0} \in \mathbb{R}^{HW}$, $v \gets \mathbf{0} \in \mathbb{R}^{N_d}$
\STATE $\gamma \gets \tau / (\tau + \bar{\epsilon})$ 
\FOR{$t = 1$ \TO $T$}
    \STATE $u \gets \bar{\epsilon} \cdot \left( - \text{LogSumExp}_{dim=2}\left(\frac{-C + v}{\bar{\epsilon}}\right) \right)$
    \STATE $v_{tgt} \gets \log(1/N_d)\bar{\epsilon} - \bar{\epsilon} \left( \text{LogSumExp}_{dim=1}\left(\frac{-C + u}{\bar{\epsilon}}\right) - \log(HW) \right)$
    \STATE $v \gets \gamma \cdot v_{tgt}$ 
\ENDFOR
\STATE $Logits \gets (-C + u + v) / \epsilon$ \quad \textit{// Apply spatial temperature}
\STATE $P \gets \exp(\text{Clamp}(Logits, \max=0))$
\STATE $P \gets P / (P.\text{sum}(dim=-1, keepdim=True) + \epsilon)$ \quad \textit{// $\epsilon$-Relaxed Unbalanced Normalization}
\RETURN $P$
\end{algorithmic}
\end{algorithm}

To enforce theoretical optimality, we apply a variance-weighted feature matching loss (Dispersion Loss) on the base structural slice:
\begin{equation}
    \mathcal{L}_{disp} = \frac{1}{HW} \sum_{i=1}^{HW} \sum_{j=1}^{N_d} P_{i,j} \| Q_i - V_j \|_2^2
\end{equation}

## 3.4. Markovian Slice-based Entropy Model with LRP
To decode the $K$ autoregressive slices, we maintain a hidden Markovian memory state $M_k \in \mathbb{R}^{C_{slice} \times H \times W}$. For the $k$-th slice, the context is formed by $c_k = [\hat{z}, M_k]$. 

To mitigate error propagation across sequential slices, we introduce **Latent Residual Prediction (LRP)**. Instead of using the raw quantized output $\hat{y}_k$ directly, we predict a continuous residual bounded by $\tanh$ to correct quantization artifacts within the valid quantization bin:
\begin{equation}
    \hat{y}_k \leftarrow \hat{y}_k + 0.5 \tanh\Big( \text{LRP\_Transform}([S_k, \hat{y}_k]) \Big)
\end{equation}