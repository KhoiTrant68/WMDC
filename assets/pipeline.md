graph TD
    %% Global Styles
    classDef data fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef model fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef wave fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,stroke-dasharray: 5 5;
    classDef dict fill:#f3e5f5,stroke:#7b1fa2,stroke-width:4px;
    classDef lrp fill:#ffe0b2,stroke:#e65100,stroke-width:3px;
    classDef loss fill:#ffcdd2,stroke:#c62828,stroke-width:2px,stroke-dasharray: 5 5;

    %% --- MAIN ENCODER BRANCH ---
    Input(Input Image x):::data --> Enc_Conv[Conv Downsampling]:::model
    Enc_Conv --> Enc_Wave1[Wavelet Residual Block 1]:::wave
    Enc_Wave1 --> Enc_Wave2[Wavelet Residual Block 2]:::wave
    Enc_Wave2 --> Enc_Conv2[Conv Downsampling]:::model
    Enc_Conv2 --> Latent_Y(Latent Representation y):::data

    %% --- THE SPECTRAL SPLIT ---
    subgraph "Spectral Split (Discrete Wavelet Transform)"
        Latent_Y --> Latent_DWT[Latent DWT 2D Transform]:::wave
        Latent_DWT -- "LL Subband (1x Channels)" --> Y_LF(Low-Freq Latent y_LF):::data
        Latent_DWT -- "LH, HL, HH Subbands (3x Channels)" --> Y_HF(High-Freq Latent y_HF):::data
    end

    %% --- HYPER-PRIOR BRANCH ---
    subgraph "Hyper-Prior Autoencoder"
        Y_LF & Y_HF --> Hyper_Enc[Hyper Encoder h_a]:::model
        Hyper_Enc --> Latent_Z(Hyper Latent z):::data
        Latent_Z --> Quant_Z[Quantize z]:::model
        Quant_Z --> Entropy_Z[Entropy Bottleneck]:::model
        Quant_Z --> Hyper_Dec[Hyper Decoder h_s]:::model
        
        Hyper_Dec -- "Spatial Means & Scales" --> Params_LF[Hyper Params LF]:::data
        Hyper_Dec -- "Spatial Means & Scales" --> Params_HF[Hyper Params HF]:::data
    end

    %% --- ASYMMETRIC ENTROPY MODEL & LRP ---
    subgraph "Asymmetric Context Modeling & Latent Residual Prediction"
        
        %% --- PATH A: LOW FREQUENCY (STRUCTURE) ---
        Y_LF --> Context_Mamba[Mamba Context Model VSSBlock]:::model
        Context_Mamba & Params_LF --> Transforms_LF[CC Transforms LF]:::model
        Transforms_LF -- "mu_lf, scale_lf" --> GMM_LF[Gaussian Conditional LF]:::model
        
        %% Encoding / Quantization
        Y_LF --> GMM_LF
        GMM_LF --> Bitstream_LF(Bits LF):::dict
        GMM_LF -- "y_lf_hat = round(y_lf - mu) + mu" --> Quant_Y_LF[Quantized y_lf_hat]:::data

        %% Latent Residual Prediction (LRP) for LF
        Quant_Y_LF & Params_LF & Context_Mamba --> LRP_Module_LF[LRP Module LF]:::lrp
        LRP_Module_LF -- "Residual" --> Add_LF((+))
        Quant_Y_LF --> Add_LF
        Add_LF --> Refined_Y_LF(Refined Latent y_lf_tilde):::data

        %% --- PATH B: HIGH FREQUENCY (TEXTURE) ---
        %% Note: HF relies on Refined LF to build its query
        Refined_Y_LF & Params_HF --> Query_Gen[Query Generator]:::model
        Learned_Dict[(Learned Global Dictionary Parameter)]:::dict <--> |Cross-Attention| Dict_Attn[Dictionary Cross-Attention Module]:::dict
        Query_Gen --> Dict_Attn
        
        Dict_Attn -- "Texture Context" --> Transforms_HF[CC Transforms HF]:::model
        Transforms_HF & Params_HF -- "mu_hf, scale_hf" --> GMM_HF[Gaussian Conditional HF]:::model
        
        %% Encoding / Quantization
        Y_HF --> GMM_HF
        GMM_HF --> Bitstream_HF(Bits HF):::dict
        GMM_HF -- "y_hf_hat = round(y_hf - mu) + mu" --> Quant_Y_HF[Quantized y_hf_hat]:::data

        %% Latent Residual Prediction (LRP) for HF
        Quant_Y_HF & Params_HF & Dict_Attn --> LRP_Module_HF[LRP Module HF]:::lrp
        LRP_Module_HF -- "Residual" --> Add_HF((+))
        Quant_Y_HF --> Add_HF
        Add_HF --> Refined_Y_HF(Refined Latent y_hf_tilde):::data

    end

    %% --- RECONSTRUCTION ---
    subgraph "Latent Merging & Decoder"
        Refined_Y_LF & Refined_Y_HF --> Latent_IDWT[Latent Inverse DWT 2D]:::wave
        Latent_IDWT --> Latent_Y_Hat(Merged Latent y_tilde):::data
        
        Latent_Y_Hat --> Dec_Upsample1[Conv Transpose]:::model
        Dec_Upsample1 --> Dec_Wave1[Wavelet Residual Upsample 1]:::wave
        Dec_Wave1 --> Dec_Wave2[Wavelet Residual Upsample 2]:::wave
        Dec_Wave2 --> Output(Reconstructed Image x_hat):::data
    end

    %% --- LOSS CALCULATION ---
    Bitstream_LF & Bitstream_HF & Entropy_Z -.-> Rate_Loss(Rate Loss: BPP):::loss
    Input & Output -.-> Dist_Loss(Distortion Loss: MSE / MS-SSIM):::loss