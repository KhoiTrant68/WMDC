graph TD
    %% Global Styles
    classDef data fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef model fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef mamba fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef wls fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px,stroke-dasharray: 5 5;
    classDef dict fill:#f3e5f5,stroke:#7b1fa2,stroke-width:4px;
    classDef routing fill:#e1bee7,stroke:#4a148c,stroke-width:4px;
    classDef lrp fill:#ffe0b2,stroke:#e65100,stroke-width:3px;
    classDef ot fill:#ffccbc,stroke:#bf360c,stroke-width:3px;
    classDef loss fill:#ffcdd2,stroke:#c62828,stroke-width:2px,stroke-dasharray: 5 5;
    classDef sliceloop fill:#fff3e0,stroke:#ef6c00,stroke-width:3px;

    %% ============================================================
    %% MAIN ENCODER  g_a  (image → y)
    %% ============================================================
    Input("Input Image x"):::data --> Enc_Down0["Conv ↓2"]:::model
    Enc_Down0 --> Enc_FDM0["FDM Block 0"]:::mamba
    Enc_FDM0 --> Enc_Down1["Conv ↓2"]:::model
    Enc_Down1 --> Enc_FDM1["FDM Block 1"]:::mamba
    Enc_FDM1 --> Enc_Down2["Conv ↓2"]:::model
    Enc_Down2 --> Enc_FDM2["FDM Block 2"]:::mamba
    Enc_FDM2 --> Enc_Down3["Conv ↓2"]:::model
    Enc_Down3 --> Latent_Y("Latent y, M channels, /16"):::data

    %% WLS multi-scale shortcuts (optional, CMIC-style)
    Input -. "WLS DWT shortcut" .-> WLS_S1["WLS shortcut 1"]:::wls
    WLS_S1 -. inject .-> Enc_Down1
    Input -. "WLS DWT shortcut" .-> WLS_S2["WLS shortcut 2"]:::wls
    WLS_S2 -. inject .-> Enc_Down2

    %% ============================================================
    %% HYPER-PRIOR  h_a → z → z_hat → hyper params
    %% ============================================================
    subgraph "Hyper-Prior Autoencoder"
        Latent_Y --> Hyper_Enc["h_a: VSSBlock + Conv ↓4"]:::model
        Hyper_Enc --> Latent_Z("Hyper Latent z"):::data
        Latent_Z --> Entropy_Z["Entropy Bottleneck"]:::model
        Entropy_Z --> Z_Hat("z hat"):::data
        Z_Hat --> H_Trunk["h_trunk"]:::model
        H_Trunk --> H_Scale["h_scale_head"]:::model
        H_Trunk --> H_Mean["h_mean_head"]:::model
        H_Scale --> Params_Sigma("latent_scales σ"):::data
        H_Mean --> Params_Mu("latent_means μ"):::data
        Params_Sigma & Params_Mu --> Hyper_Prior("hyper_prior = concat 2M ch"):::data
    end

    %% ============================================================
    %% DICTIONARY GENERATION  z_hat → learned dictionary tokens
    %% ============================================================
    subgraph "QueryDictionaryGenerator (z_hat → dt)"
        Z_Hat --> QDG_Pos["Pos Enc + Cross-Attn with learned queries"]:::dict
        QDG_Pos --> QDG_Proj["OLP Projection + L2-normalise × sqrt-d"]:::dict
        QDG_Proj --> Dict_Tokens("Dictionary tokens dt: N × dict_dim"):::dict
        QDG_Proj -. "off-diag cosine penalty" .-> Loss_DictPenalty("δ · dict_penalty"):::loss
    end

    %% ============================================================
    %% CONTENT-COMPLEXITY (anti-leakage prior)
    %% ============================================================
    Input -. detached .-> Complexity_Proxy["Sobel-based content complexity"]:::model
    Complexity_Proxy --> Complexity_Map("complexity ρ_img on z-grid"):::data

    %% ============================================================
    %% PER-SLICE AUTOREGRESSIVE LOOP  (num_slices = 5)
    %% ============================================================
    subgraph SliceLoop ["Slice-Autoregressive Loop  (i = 0 … S−1)"]
        direction TB

        %% Slice input: split y into S equal channel groups
        Latent_Y --> Slice_Split["y.chunk(S, dim=1)"]:::model
        Slice_Split --> Y_Slice_i("y_slice_i"):::data

        %% Memory state (stateful Markov mode)
        Mem_State("memory_state M_i, slice_ch"):::data
        Hyper_Prior -. "M_1 init: Conv bootstrap" .-> Mem_State

        %% Query construction
        Hyper_Prior --> Query_Cat["Concat hyper_prior ⊕ M_i"]:::model
        Mem_State --> Query_Cat
        Query_Cat --> Query("query: 2M + slice_ch"):::data

        %% Per-slice K/V projections from shared dt
        Dict_Tokens --> K_Proj["k_projs i: OLP"]:::dict
        Dict_Tokens --> V_Proj["v_projs i: OLP"]:::dict
        K_Proj --> K_i("k_i"):::data
        V_Proj --> V_i("v_i"):::data

        %% Spatial complexity mass (UEOT only)
        Hyper_Prior --> Rho_Pred["rho_predictors i: 2-layer Conv1x1"]:::model
        Complexity_Map -. "supervision via alignment hinge" .-> Loss_Align
        Rho_Pred --> Rho_Spatial("rho_spatial: B × HW"):::data

        %% =====================================================
        %% UNIFIED DICTIONARY ATTENTION  (CORE NOVELTY)
        %% =====================================================
        subgraph EOT ["UnifiedDictionaryAttention i (3 routing modes)"]
            direction TB
            Query --> Cost_Matrix["Cost C = −q_norm · k  (std≈1)"]:::routing
            K_i --> Cost_Matrix
            Cost_Matrix --> Routing{Routing mode?}:::routing

            Routing -- softmax --> P_Softmax["P = softmax(−C/τ)"]:::routing
            Routing -- balanced_eot --> P_BalSink["log-Sinkhorn: uniform a, b"]:::ot
            Routing -- unbalanced_eot --> P_UnbalSink["log-Sinkhorn: KL-relaxed marginals (a=1/HW, b conditional)"]:::ot

            Rho_Spatial --> P_UnbalSink
            Cum_Col_Usage -. log_b_override .-> P_UnbalSink
            Cum_Col_Usage -. log_b_override .-> P_BalSink

            P_Softmax --> Transport_P("Transport plan P: B × HW × N"):::routing
            P_BalSink --> Transport_P
            P_UnbalSink --> Transport_P

            Transport_P --> Dict_BMM["P @ v_norm"]:::dict
            V_i --> Dict_BMM
            Dict_BMM --> Dict_Info("dict_info_i: M ch"):::data
        end

        %% Routing-side losses (only in training)
        Transport_P -. "H_col bonus" .-> Loss_ColEntropy("β_col · −H_col, default 0"):::loss
        Transport_P -. "H_row penalty" .-> Loss_RowEntropy("β_row · H_row, default 0.3"):::loss
        Transport_P -. row_mass .-> Loss_Align("γ · ReLU κ − Pearson row_mass, ρ_img"):::loss
        Transport_P -. "spatial TV" .-> Loss_TV("tv_weight · TV(P), default 0"):::loss

        %% Update cumulative column usage for next slice (no-grad)
        Transport_P -. "col_mass detach" .-> Cum_Col_Usage("cum col usage prev slices, no-grad"):::data

        %% Context transforms → Gaussian conditional
        Query --> Support_Cat["Concat query ⊕ dict_info"]:::model
        Dict_Info --> Support_Cat
        Support_Cat --> Support("support"):::data

        Support --> CC_Mean["cc_mean_transforms i: 4-layer Conv"]:::model
        Support --> CC_Scale["cc_scale_transforms i: 4-layer Conv"]:::model
        CC_Mean --> Mu("μ_i"):::data
        CC_Scale --> Sigma("σ_i, clamp ≥ 0.11"):::data

        %% Quantisation (noise relaxation or STE)
        Y_Slice_i --> Quant_Y["Gaussian conditional: noise or round STE"]:::model
        Mu --> Quant_Y
        Sigma --> Quant_Y
        Quant_Y --> Y_Hat_Slice("y_hat_slice_i"):::data
        Quant_Y -. likelihood .-> Bits_Y_Slice("bits y_i"):::data

        %% Latent Residual Prediction
        Support --> LRP_Cat["Concat support ⊕ y_hat_slice"]:::lrp
        Y_Hat_Slice --> LRP_Cat
        LRP_Cat --> LRP_Net["lrp_transforms i: ResNet block"]:::lrp
        LRP_Net --> LRP_Residual("residual"):::data
        LRP_Residual --> LRP_Gate["× softplus lrp_scales i"]:::lrp
        LRP_Gate --> LRP_Add((+))
        Y_Hat_Slice --> LRP_Add
        LRP_Add --> Y_Hat_Slice_LRP("y_hat_slice_i refined"):::data

        %% Memory update for next slice
        Y_Hat_Slice_LRP --> Mem_Upd["memory_updaters i: GatedMemoryUpdater"]:::model
        Mem_State --> Mem_Upd
        Mem_Upd -. "M_i+1 to next slice" .-> Mem_State
    end

    %% ============================================================
    %% MERGE SLICES → MAIN DECODER  g_s  (y_hat → x_hat)
    %% ============================================================
    subgraph "Decoder g_s"
        Y_Hat_Slice_LRP --> Slice_Merge["Concat over slices"]:::model
        Slice_Merge --> Y_Hat("y hat: M ch"):::data
        Y_Hat --> Dec_Up0["ConvTranspose ↑2"]:::model
        Dec_Up0 --> Dec_FDM0["FDM Block 0"]:::mamba
        Dec_FDM0 --> Dec_Up1["ConvTranspose ↑2"]:::model
        Dec_Up1 --> Dec_FDM1["FDM Block 1"]:::mamba
        Dec_FDM1 --> Dec_Up2["ConvTranspose ↑2"]:::model
        Dec_Up2 --> Dec_FDM2["FDM Block 2"]:::mamba
        Dec_FDM2 --> Dec_Up3["ConvTranspose ↑2"]:::model
        Dec_Up3 --> Output("Reconstructed Image x_hat"):::data

        %% iWLS multi-scale injection
        Y_Hat -. "iWLS shortcut" .-> WLS_iS1["iWLS shortcut 1"]:::wls
        WLS_iS1 -. inject .-> Dec_Up1
        Y_Hat -. "iWLS shortcut" .-> WLS_iS2["iWLS shortcut 2"]:::wls
        WLS_iS2 -. inject .-> Dec_Up2
    end

    %% ============================================================
    %% LOSS AGGREGATION
    %% ============================================================
    Bits_Y_Slice & Entropy_Z -.-> Rate_Loss("R = BPP from likelihoods"):::loss
    Input & Output -.-> Dist_Loss("D = MSE or 1 − MS-SSIM"):::loss
    Rate_Loss --> Total_Loss(("L_total = λ·D + R + losses")):::loss
    Dist_Loss --> Total_Loss
    Loss_ColEntropy --> Total_Loss
    Loss_RowEntropy --> Total_Loss
    Loss_Align --> Total_Loss
    Loss_DictPenalty --> Total_Loss
    Loss_TV --> Total_Loss

    %% Style the slice loop subgraph
    class SliceLoop sliceloop